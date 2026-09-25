from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np


@dataclass
class STTSegment:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


class BaseSTTEngine(ABC):
    """Abstract provider interface for all Speech-to-Text backends."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.task = task
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights into GPU/CPU memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights and free GPU memory."""
        pass

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> str:
        """Transcribe a single short audio segment to clean text."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> list[STTSegment]:
        """Transcribe an entire audio array and return timestamped segments."""
        pass

    def transcribe_file(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> list[STTSegment]:
        """Transcribe an audio or video file from disk."""
        from autocut.cut import extract_audio_from_video
        audio = extract_audio_from_video(file_path, sample_rate=16000)
        return self.transcribe(audio, sample_rate=16000, **kwargs)
