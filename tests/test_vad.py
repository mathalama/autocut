"""Stage 1 contract tests for the bundled faster-whisper VAD."""

from pathlib import Path

import numpy as np

from autocut.stages.vad import detect_speech


def test_detect_speech_converts_16khz_samples_to_seconds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("autocut.stages.vad.decode_audio", lambda *_args, **_kwargs: np.zeros(32000))
    monkeypatch.setattr(
        "autocut.stages.vad.get_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 1600, "end": 8000}],
    )

    output = tmp_path / "speech.json"
    result = detect_speech(tmp_path / "audio.wav", output)

    assert [(interval.start, interval.end) for interval in result.intervals] == [(0.1, 0.5)]
    assert output.read_text(encoding="utf-8").startswith('{\n  "source"')
