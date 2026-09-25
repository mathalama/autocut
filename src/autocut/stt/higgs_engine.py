from __future__ import annotations

import gc
import logging
from typing import Any, Optional
import numpy as np

from autocut.stt.base import BaseSTTEngine, STTSegment
from autocut.engine import clean_hallucinations

logger = logging.getLogger("autocut.stt.higgs")


class HiggsEngine(BaseSTTEngine):
    """STT provider backed by Boson AI Higgs Audio v3 STT (LLM decoder)."""

    def __init__(
        self,
        model_name: str = "bosonai/higgs-audio-v3-stt",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> None:
        super().__init__(model_name, device, compute_type, language, task)
        self._wrapper = None

    def load(self) -> None:
        if self._is_loaded and self._wrapper is not None:
            return

        from autocut.higgs import get_higgs_model
        torch_dtype = "float16" if self.compute_type in ("float16", "fp16") else "float32"
        self._wrapper = get_higgs_model(
            model_id=self.model_name,
            device=self.device,
            torch_dtype=torch_dtype,
        )
        self._is_loaded = True

    def unload(self) -> None:
        if self._wrapper is not None:
            del self._wrapper
            self._wrapper = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

        raw = self._wrapper.transcribe(audio_norm, sample_rate=sample_rate, language=lang)
        raw_text = " ".join(raw) if isinstance(raw, list) else str(raw)
        cleaned = clean_hallucinations(raw_text)
        return cleaned

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

        import torch
        audio_norm = audio.astype(np.float32)
        lang = None if self.language in (None, "", "auto") else self.language

        # Segment using Silero VAD into distinct speech intervals
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        get_speech_timestamps = utils[0]

        tensor_audio = torch.from_numpy(audio_norm)
        speech_timestamps = get_speech_timestamps(
            tensor_audio,
            vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=400,
            speech_pad_ms=150,
        )

        results: list[STTSegment] = []
        idx = 1

        if not speech_timestamps:
            # Fallback if no specific timestamps were segmented
            raw = self._wrapper.transcribe(audio_norm, sample_rate=sample_rate, language=lang)
            raw_text = " ".join(raw) if isinstance(raw, list) else str(raw)
            cleaned = clean_hallucinations(raw_text)
            if cleaned:
                results.append(STTSegment(
                    index=1,
                    start=0.0,
                    end=len(audio_norm) / sample_rate,
                    text=cleaned,
                ))
            return results

        for ts in speech_timestamps:
            chunk = audio_norm[ts["start"]:ts["end"]]
            if len(chunk) < int(sample_rate * 0.25):
                continue

            raw = self._wrapper.transcribe(chunk, sample_rate=sample_rate, language=lang)
            raw_text = " ".join(raw) if isinstance(raw, list) else str(raw)
            cleaned = clean_hallucinations(raw_text)
            if cleaned:
                start_s = ts["start"] / sample_rate
                end_s = ts["end"] / sample_rate
                results.append(STTSegment(
                    index=idx,
                    start=round(start_s, 2),
                    end=round(end_s, 2),
                    text=cleaned,
                ))
                idx += 1

        return results
