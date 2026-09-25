from __future__ import annotations

import gc
import logging
from typing import Any, Optional
import numpy as np

from autocut.stt.base import BaseSTTEngine, STTSegment
from autocut.engine import register_cuda_dlls, clean_hallucinations

logger = logging.getLogger("autocut.stt.whisper")


class WhisperEngine(BaseSTTEngine):
    """STT provider backed by faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> None:
        super().__init__(model_name, device, compute_type, language, task)
        self._model = None

    def load(self) -> None:
        if self._is_loaded and self._model is not None:
            return

        register_cuda_dlls()
        from faster_whisper import WhisperModel

        actual_device = self.device
        actual_compute = self.compute_type

        if actual_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    actual_device = "cpu"
                    actual_compute = "int8"
            except Exception:
                actual_device = "cpu"
                actual_compute = "int8"

        try:
            self._model = WhisperModel(
                self.model_name,
                device=actual_device,
                compute_type=actual_compute,
            )
            self._is_loaded = True
            self.device = actual_device
            self.compute_type = actual_compute
        except Exception as e:
            if actual_device == "cuda":
                logger.warning("CUDA initialization failed (%s). Falling back to CPU.", e)
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self._is_loaded = True
                self.device = "cpu"
                self.compute_type = "int8"
            else:
                raise

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False
        gc.collect()

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> str:
        if not self._is_loaded:
            self.load()

        if len(audio_chunk) < int(sample_rate * 0.3):
            return ""

        audio_norm = audio_chunk.astype(np.float32)
        lang = None if self.language in (None, "", "auto") else self.language

        segments, info = self._model.transcribe(
            audio_norm,
            language=lang,
            task=self.task,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        texts = []
        for s in segments:
            cleaned = clean_hallucinations(s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio)
            if cleaned:
                texts.append(cleaned)

        return " ".join(texts).strip()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> list[STTSegment]:
        if not self._is_loaded:
            self.load()

        if len(audio) < int(sample_rate * 0.3):
            return []

        audio_norm = audio.astype(np.float32)
        lang = None if self.language in (None, "", "auto") else self.language
        beam_size = kwargs.get("beam_size", 5)

        raw_segments, info = self._model.transcribe(
            audio_norm,
            language=lang,
            task=self.task,
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=150),
            condition_on_previous_text=False,
        )

        results: list[STTSegment] = []
        idx = 1
        for s in raw_segments:
            cleaned = clean_hallucinations(s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio)
            if cleaned:
                results.append(STTSegment(
                    index=idx,
                    start=float(s.start),
                    end=float(s.end),
                    text=cleaned,
                ))
                idx += 1

        return results
