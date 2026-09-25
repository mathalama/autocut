from unittest.mock import Mock, patch
import numpy as np
import pytest

from autocut.stt.base import BaseSTTEngine, STTSegment
from autocut.stt.pool import ModelPool


def test_stt_segment_to_dict():
    seg = STTSegment(index=1, start=0.5, end=2.3, text="Hello world")
    d = seg.to_dict()
    assert d["index"] == 1
    assert d["start"] == 0.5
    assert d["end"] == 2.3
    assert d["text"] == "Hello world"


class DummyEngine(BaseSTTEngine):
    def load(self):
        self._is_loaded = True

    def unload(self):
        self._is_loaded = False

    def transcribe_chunk(self, audio_chunk, sample_rate=16000, **kwargs):
        return "dummy text"

    def transcribe(self, audio, sample_rate=16000, **kwargs):
        return [STTSegment(index=1, start=0.0, end=1.0, text="dummy text")]


def test_model_pool_singleton():
    pool1 = ModelPool.instance()
    pool2 = ModelPool.instance()
    assert pool1 is pool2


def test_model_pool_lifecycle():
    pool = ModelPool.instance()
    dummy = DummyEngine("test_model", device="cpu")

    with patch("autocut.stt.pool.WhisperEngine", return_value=dummy):
        engine = pool.get_engine("whisper", "test_model", device="cpu")
        assert engine.is_loaded
        assert pool._current_engine is engine

        pool.unload_all()
        assert not dummy.is_loaded
        assert pool._current_engine is None
