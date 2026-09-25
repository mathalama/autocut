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


def register_cuda_dlls() -> None:
    """Register NVIDIA site-packages and torch/lib DLL directories and preload CUDA libraries on Windows."""
    if sys.platform != "win32":
        return
    import ctypes
    search_dirs = []
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        internal_dir = base_dir / "_internal"
        search_dirs.extend([
            base_dir,
            internal_dir,
            internal_dir / "torch" / "lib",
        ])
        if internal_dir.is_dir():
            for p in internal_dir.glob("nvidia/*/bin"):
                search_dirs.append(p)
    else:
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
        search_dirs.append(site_packages / "torch" / "lib")
        for nvidia_bin in site_packages.glob("nvidia/*/bin"):
            search_dirs.append(nvidia_bin)

    for d in search_dirs:
        if d.is_dir():
            resolved = str(d.resolve())
            try:
                _DLL_HANDLES.append(os.add_dll_directory(resolved))
            except Exception:
                pass
            os.environ["PATH"] = resolved + ";" + os.environ.get("PATH", "")


register_cuda_dlls()

from faster_whisper import WhisperModel
from autocut.audio import AudioStreamer
from autocut.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    FADE_OUT_SECONDS,
    MAX_SPEECH_DURATION,
    MIN_SPEECH_DURATION,
    SAMPLE_RATE,
    SILENCE_FINALIZE_SECONDS,
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

    if "live conversation clean subtitles" in normalized:
        return ""

    # Check for pathological word repetition (e.g., "you you you you")
    words = normalized.split()
    if len(words) >= 4 and len(set(words)) == 1:
        logger.debug("Filtered repetition loop: '%s'", text)
        return ""

    return raw


class RealtimeSubtitleEngine:
    """Production-grade subtitle engine with Single-Pass Final Transcription and modular STT backends."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = DEFAULT_LANGUAGE,
        task: str = "transcribe",
        engine_type: str = "whisper",
        device_index: Optional[int] = None,
        energy_threshold: float = 0.015,
        on_caption: Optional[Callable[[str, bool], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_status: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.task = task
        self.engine_type = engine_type.lower()
        # Translation requires a multilingual model (cannot use English-only .en models)
        if self.task == "translate" and model_name.endswith(".en"):
            fallback = model_name[:-3]
            logger.info("Translation requires multilingual model. Switching '%s' to '%s'.", model_name, fallback)
            model_name = fallback

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.device_index = device_index
        self.energy_threshold = energy_threshold
        self.on_caption = on_caption
        self.on_clear = on_clear
        self.on_level = on_level
        self.on_status = on_status

        self.vad = VoiceActivityDetector(energy_threshold=self.energy_threshold)
        self.streamer = AudioStreamer(device_index=self.device_index)
        self.model: Any = None

        # Concurrency & Locks
        self._inference_lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def pause(self) -> None:
        """Pause subtitle capture and hide overlay."""
        self._paused = True
        if self.on_clear:
            self.on_clear()
        if self.on_status:
            self.on_status({"paused": True})

    def resume(self) -> None:
        """Resume subtitle capture."""
        self._paused = False
        if self.on_status:
            self.on_status({"paused": False})

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns True if now paused, False if resumed."""
        if self._paused:
            self.resume()
            return False
        else:
            self.pause()
            return True

    @property
    def is_paused(self) -> bool:
        return self._paused

    def load_model(self) -> None:
        """Load the configured STT model (Whisper or Higgs Audio v3)."""
        if self.engine_type == "higgs":
            from autocut.higgs import HiggsSTTModel
            logger.info("Loading Higgs Audio v3 STT model '%s'...", self.model_name)
            self.model = HiggsSTTModel(
                model_id=self.model_name if "higgs" in self.model_name else "bosonai/higgs-audio-v3-stt",
                device=self.device if "cuda" in self.device else "cuda:0",
            )
            return

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
            list(self.model.transcribe(dummy, language=lang, task=self.task, beam_size=1)[0])
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

    def _transcribe_audio(self, audio_window: np.ndarray, active_language: Optional[str]) -> str:
        """Transcribe an audio segment directly in one pass."""
        if self.engine_type == "higgs":
            results = self.model.transcribe(audio_window, language=active_language)
            return " ".join(results).strip()

        segments, _ = self.model.transcribe(
            audio_window,
            language=active_language,
            task=self.task,
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
        return " ".join(valid_parts).strip()

    def _worker(self) -> None:
        """Single-pass utterance listener and transcription loop.
        
        Eliminates intermediate draft flickering and double-inference:
        Speech is detected -> phrase ends -> transcribed ONCE directly -> emitted as final.
        """
        last_speech_time = time.time()
        speech_start_time = 0.0
        is_speaking = False
        fade_sent = False
        active_language = self.language if self.language and self.language != "auto" else None

        while self._running:
            time.sleep(0.02)  # 50Hz polling loop
            if self._paused:
                if is_speaking:
                    is_speaking = False
                continue

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
                    fade_sent = False
                last_speech_time = now

            if is_speaking:
                speech_duration = now - speech_start_time
                silence_duration = now - last_speech_time

                # 1) Natural pause after speaking
                if silence_duration >= SILENCE_FINALIZE_SECONDS:
                    actual_speech_len = last_speech_time - speech_start_time
                    if actual_speech_len >= MIN_SPEECH_DURATION:
                        audio_window = self.streamer.get_recent_audio(min(actual_speech_len + 0.20, 15.0))
                        if len(audio_window) > 0 and self._inference_lock.acquire(blocking=True, timeout=0.5):
                            try:
                                raw_text = self._transcribe_audio(audio_window, active_language)
                                cleaned = clean_hallucinations(raw_text, calculate_rms(audio_window))
                                if cleaned and self.on_caption:
                                    self.on_caption(cleaned, True)
                            except Exception as e:
                                logger.debug("Inference error: %s", e)
                            finally:
                                self._inference_lock.release()
                    # In all cases when silence reached, reset is_speaking
                    is_speaking = False

                # 2) Continuous monologue reached max chunk duration
                elif speech_duration >= MAX_SPEECH_DURATION:
                    audio_window = self.streamer.get_recent_audio(min(speech_duration + 0.15, 15.0))
                    if len(audio_window) > 0 and self._inference_lock.acquire(blocking=True, timeout=0.5):
                        try:
                            raw_text = self._transcribe_audio(audio_window, active_language)
                            cleaned = clean_hallucinations(raw_text, calculate_rms(audio_window))
                            if cleaned and self.on_caption:
                                self.on_caption(cleaned, True)
                        except Exception as e:
                            logger.debug("Inference error: %s", e)
                        finally:
                            self._inference_lock.release()
                    speech_start_time = now
                    last_speech_time = now

            # Fade out overlay on prolonged silence
            silence_duration = now - last_speech_time
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
