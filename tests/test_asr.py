"""Stage 1 contract tests for timestamped transcription serialization."""

from pathlib import Path
from types import SimpleNamespace

from autocut.models import SpeechInterval
from autocut.stages.asr import transcribe


def test_transcribe_serializes_word_timestamps_and_selected_intervals(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured["model_init"] = kwargs

        def transcribe(self, _audio, **kwargs):
            captured["transcribe"] = kwargs
            words = [SimpleNamespace(word=" Codex", start=1.0, end=1.3, probability=0.9)]
            segment = SimpleNamespace(start=1.0, end=1.3, text=" Codex", words=words)
            return iter([segment]), SimpleNamespace(language="ru", language_probability=0.99)

    monkeypatch.setattr("autocut.stages.asr.WhisperModel", FakeModel)
    output = tmp_path / "transcript.json"

    transcript = transcribe(
        audio_path=tmp_path / "audio.wav",
        intervals=[SpeechInterval(start=1.0, end=2.0)],
        output_path=output,
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        language="ru",
        initial_prompt="Codex",
    )

    assert captured["model_init"] == {
        "model_size_or_path": "large-v3-turbo", "device": "cuda", "compute_type": "float16"
    }
    assert captured["transcribe"]["clip_timestamps"] == [1.0, 2.0]
    assert captured["transcribe"]["word_timestamps"] is True
    assert transcript.segments[0].words[0].word == "Codex"
    assert output.exists()
