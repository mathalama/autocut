from __future__ import annotations

import logging
import threading
from typing import Optional

from autocut.stt.base import BaseSTTEngine
from autocut.stt.whisper_engine import WhisperEngine
from autocut.stt.higgs_engine import HiggsEngine

logger = logging.getLogger("autocut.stt.pool")


class ModelPool:
    """Thread-safe VRAM and STT engine lifecycle manager."""

    _instance: Optional[ModelPool] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._current_engine: Optional[BaseSTTEngine] = None
        self._current_key: Optional[str] = None
        self._engine_lock = threading.Lock()

    @classmethod
    def instance(cls) -> ModelPool:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_engine(
        self,
        engine_type: str = "whisper",
        model_name: Optional[str] = None,
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> BaseSTTEngine:
        """Retrieve an active engine instance, unloading any conflicting engine to conserve VRAM."""
        normalized_type = engine_type.lower().strip()
        if model_name is None:
            model_name = "bosonai/higgs-audio-v3-stt" if normalized_type == "higgs" else "small.en"

        key = f"{normalized_type}::{model_name}::{device}::{compute_type}::{language}::{task}"

        with self._engine_lock:
            if self._current_engine is not None:
                if self._current_key == key and self._current_engine.is_loaded:
                    return self._current_engine

                logger.info("Switching STT engine from %s to %s. Releasing VRAM.", self._current_key, key)
                self._current_engine.unload()
                self._current_engine = None
                self._current_key = None

            if normalized_type == "higgs":
                engine = HiggsEngine(
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                    task=task,
                )
            else:
                engine = WhisperEngine(
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                    task=task,
                )

            engine.load()
            self._current_engine = engine
            self._current_key = key
            return engine

    def unload_all(self) -> None:
        """Unload active engine and free all GPU/CPU resources."""
        with self._engine_lock:
            if self._current_engine is not None:
                logger.info("Unloading engine: %s", self._current_key)
                self._current_engine.unload()
                self._current_engine = None
                self._current_key = None
