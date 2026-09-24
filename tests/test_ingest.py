"""Stage 1 contract tests for media probing and WAV extraction."""

from pathlib import Path

from autocut.stages.ingest import extract_audio, probe_media


def test_probe_media_reads_duration_from_ffprobe_json(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "recording.mkv"
    source.touch()

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = '{"format": {"duration": "12.5"}}'

        return Result()

    monkeypatch.setattr("autocut.stages.ingest.subprocess.run", fake_run)

    assert probe_media(source).duration == 12.5


def test_extract_audio_writes_standardized_wav_command(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "recording.mp4"
    target = tmp_path / "work" / "audio.wav"
    calls = []

    monkeypatch.setattr(
        "autocut.stages.ingest.subprocess.run",
        lambda command, **_kwargs: calls.append(command),
    )

    assert extract_audio(source, target) == target
    assert calls == [[
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(target),
    ]]
