"""High-accuracy offline/batch transcription engine for AutoCut.

Processes full audio recordings or video files with full-context speech recognition
using either Whisper (e.g. large-v3-turbo, small) or Higgs Audio v3 STT.
"""

import logging
from pathlib import Path
import tempfile
from typing import Any, Optional
import numpy as np

from autocut.cut import extract_audio_16k
from autocut.recorder import CaptionRecorder

logger = logging.getLogger(__name__)


def load_wav_as_float32(wav_path: Path) -> np.ndarray:
    """Load a 16kHz mono WAV file into float32 numpy array."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
        else:
            data = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32) / 128.0

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        return data


def save_float32_as_wav(audio: np.ndarray, wav_path: Path, sample_rate: int = 16000) -> None:
    """Save float32 audio numpy array to 16-bit PCM WAV."""
    import wave
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


class BatchTranscriber:
    """High-accuracy batch speech transcriber with timestamped subtitle generation."""

    def __init__(
        self,
        engine_type: str = "whisper",
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "auto",
    ) -> None:
        self.engine_type = engine_type.lower()
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = None if (not language or language == "auto") else language
        self.model: Any = None
        self._load()

    def _load(self) -> None:
        if self.engine_type == "higgs":
            from autocut.higgs import HiggsSTTModel
            logger.info("Loading Higgs Audio v3 model '%s'...", self.model_name)
            self.model = HiggsSTTModel(
                model_id=self.model_name if "higgs" in self.model_name else "bosonai/higgs-audio-v3-stt",
                device=self.device if "cuda" in self.device else "cuda:0",
            )
        else:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model '%s' on %s...", self.model_name, self.device)
            try:
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                if self.device == "cuda":
                    logger.warning("CUDA loading failed (%s). Falling back to CPU int8...", e)
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                else:
                    raise e

    def transcribe_file(
        self,
        file_path: Path,
        srt_path: Optional[Path] = None,
        txt_path: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Transcribe an audio or video file by extracting audio first."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_wav = Path(tmp_dir) / "extracted_16k.wav"
            extract_audio_16k(file_path, tmp_wav)
            audio = load_wav_as_float32(tmp_wav)
            return self.transcribe(audio, srt_path=srt_path, txt_path=txt_path)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        srt_path: Optional[Path] = None,
        txt_path: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Transcribe audio array into timestamped subtitle segments."""
        if len(audio) == 0:
            return []

        segments_out: list[dict[str, Any]] = []
        recorder = CaptionRecorder(srt_path=srt_path, txt_path=txt_path) if (srt_path or txt_path) else None

        if self.engine_type == "higgs":
            # For Higgs, segment audio using Silero VAD into full speech chunks and transcribe each
            from faster_whisper.vad import get_vad_model, VadOptions
            vad_model = get_vad_model()
            speech_chunks = vad_model.generate_segments(
                audio,
                vad_parameters=VadOptions(
                    min_speech_duration_ms=250,
                    max_speech_duration_s=25,
                    min_silence_duration_ms=400,
                    threshold=0.45,
                )
            )

            idx = 1
            for chunk in speech_chunks:
                start_sec = chunk["start"] / sample_rate
                end_sec = chunk["end"] / sample_rate
                audio_slice = audio[chunk["start"] : chunk["end"]]
                if len(audio_slice) < int(0.2 * sample_rate):
                    continue

                text_list = self.model.transcribe(audio_slice, language=self.language)
                text = " ".join(text_list).strip()
                if text:
                    seg = {"index": idx, "start": start_sec, "end": end_sec, "text": text}
                    segments_out.append(seg)
                    if recorder:
                        recorder.add_caption(text, start_sec, end_sec)
                    idx += 1

            # Fallback if VAD didn't produce chunks but audio exists
            if not segments_out and len(audio) > sample_rate:
                text_list = self.model.transcribe(audio, language=self.language)
                text = " ".join(text_list).strip()
                if text:
                    dur = len(audio) / sample_rate
                    seg = {"index": 1, "start": 0.0, "end": dur, "text": text}
                    segments_out.append(seg)
                    if recorder:
                        recorder.add_caption(text, 0.0, dur)

        else:
            # Whisper with beam_size=5 for maximum precision
            whisper_segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_speech_duration_ms=200, threshold=0.40),
            )

            idx = 1
            for seg in whisper_segments:
                clean_text = seg.text.strip()
                if clean_text:
                    item = {
                        "index": idx,
                        "start": seg.start,
                        "end": seg.end,
                        "text": clean_text,
                    }
                    segments_out.append(item)
                    if recorder:
                        recorder.add_caption(clean_text, seg.start, seg.end)
                    idx += 1

        return segments_out
