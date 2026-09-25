import time
import numpy as np
from autocut.stt.rolling import RollingTranscriber
from autocut.stt.base import BaseSTTEngine, STTSegment


class MockEngine(BaseSTTEngine):
    def load(self):
        self._is_loaded = True

    def unload(self):
        self._is_loaded = False

    def transcribe_chunk(self, audio_chunk, sample_rate=16000, **kwargs):
        return "recognized phrase"

    def transcribe(self, audio, sample_rate=16000, **kwargs):
        return [STTSegment(index=1, start=0.0, end=1.0, text="recognized phrase")]


def test_rolling_transcriber_lifecycle():
    engine = MockEngine("mock")
    emitted = []

    rolling = RollingTranscriber(
        engine=engine,
        sample_rate=16000,
        pause_threshold=0.2,
        max_chunk_sec=1.0,
        min_speech_sec=0.2,
        energy_threshold=0.01,
        on_segment=lambda s: emitted.append(s),
    )
    rolling.start()

    # Feed voice frames (sine wave with audible amplitude)
    t = np.linspace(0, 0.4, int(16000 * 0.4), endpoint=False)
    voice_chunk = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rolling.process_audio_chunk(voice_chunk)

    # Feed silence to trigger pause threshold
    silent_chunk = np.zeros(int(16000 * 0.3), dtype=np.float32)
    rolling.process_audio_chunk(silent_chunk)

    time.sleep(0.3)
    results = rolling.finish()

    assert len(results) >= 1
    assert results[0].text == "recognized phrase"
    assert results[0].index == 1
