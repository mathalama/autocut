"""Resilient real-time streaming transcription engine."""

import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Optional, Set
import numpy as np

# On Windows, register NVIDIA CUDA site-packages bin directories and preload DLLs before ctranslate2/faster_whisper loads
_DLL_HANDLES = []
if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for nvidia_bin in site_packages.glob("nvidia/*/bin"):
        if nvidia_bin.is_dir():
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(nvidia_bin.resolve())))
                os.environ["PATH"] = str(nvidia_bin.resolve()) + ";" + os.environ.get("PATH", "")
            except Exception:
                pass
    import ctypes
    for dll in site_packages.glob("nvidia/*/bin/*.dll"):
        try:
            _DLL_HANDLES.append(ctypes.CDLL(str(dll.resolve())))
        except Exception:
            pass

from faster_whisper import WhisperModel
from autocut.audio import AudioStreamer
from autocut.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    FADE_OUT_SECONDS,
    MIN_SPEECH_DURATION,
    SAMPLE_RATE,
    SILENCE_FINALIZE_SECONDS,
    STEP_SECONDS,
)
from autocut.vad import VoiceActivityDetector, calculate_rms

logger = logging.getLogger(__name__)

# Known Whisper phantom hallucination phrases during silence/low signal
GHOST_HALLUCINATIONS: Set[str] = {
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "please subscribe",
    "subscribe to my channel",
    "see you next time",
    "the end",
    "i'm sorry",
    "i hope that was good",
    "do you mean we have the last one",
    "you need to get the rest of your vision please",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "ставьте лайки",
    "до скорых встреч",
    "до свидания",
    "продолжение следует",
    "конец",
}


def is_valid_segment(segment: Any) -> bool:
    """Strict confidence gate: reject any hallucination or low-probability guess."""
    # 1. Reject if no_speech_prob is elevated
    if getattr(segment, "no_speech_prob", 0.0) > 0.60:
        return False
    # 2. Reject if average log probability is low (whisper hallucination/mumbling is < -1.20)
    if getattr(segment, "avg_logprob", 0.0) < -1.20:
        return False
    # 3. Reject if repetition loops are detected via high compression ratio
    if getattr(segment, "compression_ratio", 1.0) > 2.4:
        return False
    return True


def clean_hallucinations(text: str, rms_energy: float) -> str:
    """Filter out known phantom hallucination phrases and repetition loops."""
    raw = text.strip()
    normalized = re.sub(r"[^\w\s]+", "", raw.casefold(), flags=re.UNICODE).strip()
    if not normalized:
        return ""

    # Check if phrase is a known ghost hallucination during weak audio
    if normalized in GHOST_HALLUCINATIONS and rms_energy < 0.015:
        logger.debug("Filtered ghost hallucination: '%s' (RMS: %.4f)", text, rms_energy)
        return ""

    # Check for pathological word repetition (e.g., "you you you you")
    words = normalized.split()
    if len(words) >= 4 and len(set(words)) == 1:
        logger.debug("Filtered repetition loop: '%s'", text)
        return ""

    return raw


class RealtimeSubtitleEngine:
    """Production-grade subtitle engine with Inference Gate, Anti-Hallucination, and Dynamic Throttling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = DEFAULT_LANGUAGE,
        device_index: Optional[int] = None,
        energy_threshold: float = 0.015,
        on_caption: Optional[Callable[[str, bool], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.device_index = device_index
        self.energy_threshold = energy_threshold
        self.on_caption = on_caption
        self.on_clear = on_clear
        self.on_level = on_level

        self.vad = VoiceActivityDetector(energy_threshold=self.energy_threshold)
        self.streamer = AudioStreamer(device_index=self.device_index)
        self.model: Optional[WhisperModel] = None

        # Concurrency & Adaptive Throttling
        self._inference_lock = threading.Lock()
        self._adaptive_step = STEP_SECONDS
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def load_model(self) -> None:
        """Load the faster-whisper model on configured device (with automatic CPU fallback on missing DLLs)."""
        logger.info("Loading Whisper model '%s' on %s...", self.model_name, self.device)
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            # Warm-up check: verify CUDA libraries (cublas, cudnn) actually work
            dummy = np.zeros(1600, dtype=np.float32)
            lang = self.language if self.language and self.language != "auto" else None
            list(self.model.transcribe(dummy, language=lang, beam_size=1)[0])
        except Exception as e:
            if self.device == "cuda":
                logger.warning("CUDA execution failed (%s). Falling back to high-performance CPU (int8)...", e)
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise e

    def _worker(self) -> None:
        last_speech_time = time.time()
        last_inference_time = 0.0
        current_text = ""
        is_speaking = False
        fade_sent = False
        speech_start_time = 0.0
        active_language = self.language if self.language and self.language != "auto" else None

        while self._running:
            time.sleep(0.02)  # 50Hz polling loop
            now = time.time()

            # Read recent 100ms slice
            recent_chunk = self.streamer.get_recent_audio(0.1)
            rms = calculate_rms(recent_chunk)
            speech_active = self.vad.is_speech_active(recent_chunk)

            if self.on_level and rms > 0.001:
                self.on_level(rms)

            if speech_active:
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = now
                    active_language = self.language if self.language and self.language != "auto" else None
                last_speech_time = now
                fade_sent = False

            current_speech_dur = (now - speech_start_time) if is_speaking else 0.0

            # Step interval check with adaptive throttling
            if (
                is_speaking
                and current_speech_dur >= MIN_SPEECH_DURATION
                and (now - last_inference_time) >= self._adaptive_step
            ):
                # Non-blocking inference gate (guarantees zero queue buildup on GPU)
                if self._inference_lock.acquire(blocking=False):
                    try:
                        last_inference_time = now
                        audio_window = self.streamer.get_recent_audio(min(current_speech_dur + 0.15, 12.0))
                        window_rms = calculate_rms(audio_window)

                        t0 = time.perf_counter()
                        segments, info = self.model.transcribe(
                            audio_window,
                            language=active_language,
                            beam_size=1,
                            no_speech_threshold=0.55,
                            condition_on_previous_text=False,
                            vad_filter=True,
                            vad_parameters=dict(min_speech_duration_ms=100, threshold=0.40),
                        )
                        # Lock detected language for current utterance to eliminate re-detection latency
                        if active_language is None and getattr(info, "language", None):
                            active_language = info.language

                        valid_parts = [
                            seg.text.strip()
                            for seg in segments
                            if is_valid_segment(seg) and seg.text.strip()
                        ]
                        raw_text = " ".join(valid_parts).strip()
                        inference_time = time.perf_counter() - t0

                        # Dynamic Throttling
                        if inference_time > 0.180:
                            self._adaptive_step = min(0.40, self._adaptive_step * 1.15)
                        elif inference_time < 0.080:
                            self._adaptive_step = max(STEP_SECONDS, self._adaptive_step * 0.95)

                        cleaned_text = clean_hallucinations(raw_text, window_rms)
                        if cleaned_text:
                            current_text = cleaned_text
                            if self.on_caption:
                                self.on_caption(current_text, False)
                    except Exception as e:
                        logger.debug("Inference error: %s", e)
                    finally:
                        self._inference_lock.release()

            # Silence finalize check
            silence_duration = now - last_speech_time
            if is_speaking and silence_duration >= SILENCE_FINALIZE_SECONDS:
                if self._inference_lock.acquire(blocking=True, timeout=0.3):
                    try:
                        audio_window = self.streamer.get_recent_audio(min(current_speech_dur + 0.2, 12.0))
                        segments, _ = self.model.transcribe(
                            audio_window,
                            language=active_language,
                            beam_size=1,
                            no_speech_threshold=0.55,
                            condition_on_previous_text=False,
                            vad_filter=True,
                            vad_parameters=dict(min_speech_duration_ms=100, threshold=0.40),
                        )
                        valid_parts = [
                            seg.text.strip()
                            for seg in segments
                            if is_valid_segment(seg) and seg.text.strip()
                        ]
                        final_text = " ".join(valid_parts).strip()
                        cleaned = clean_hallucinations(final_text, calculate_rms(audio_window))
                        if cleaned:
                            current_text = cleaned
                    except Exception:
                        pass
                    finally:
                        self._inference_lock.release()

                if current_text and self.on_caption:
                    self.on_caption(current_text, True)
                current_text = ""
                is_speaking = False
                active_language = self.language if self.language and self.language != "auto" else None

            # Fade out overlay on prolonged silence
            if not is_speaking and not fade_sent and silence_duration >= FADE_OUT_SECONDS:
                fade_sent = True
                if self.on_clear:
                    self.on_clear()

    def start(self) -> None:
        if self._running:
            return
        if self.model is None:
            self.load_model()
        self._running = True
        self.streamer.start()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.streamer.stop()
        if self._thread:
            self._thread.join(timeout=1.0)
