"""Stage 1 and Stage 2 CLI contract tests."""

from pathlib import Path

from typer.testing import CliRunner

from autocut.cli import app


def test_cli_exposes_stage_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "vad" in result.stdout
    assert "transcribe" in result.stdout
    assert "analyze" in result.stdout
    assert "screen" in result.stdout
    assert "subtitles" in result.stdout
    assert "render" in result.stdout
    assert "listen-markers" in result.stdout
    assert "run" in result.stdout


def test_cli_analyze_command(tmp_path: Path) -> None:
    from autocut.models import SpeechInterval, SpeechIntervals, Transcript, TranscriptSegment, Word
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    speech = SpeechIntervals(source="audio.wav", intervals=[SpeechInterval(start=0.0, end=2.0)])
    transcript = Transcript(
        source="audio.wav",
        model="test",
        language="ru",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="это типа важно",
                words=[
                    Word(word="это", start=0.0, end=0.2),
                    Word(word="типа", start=0.5, end=0.7),
                    Word(word="важно", start=0.75, end=1.0),
                ],
            )
        ],
    )
    (work_dir / "speech.json").write_text(speech.model_dump_json(), encoding="utf-8")
    (work_dir / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(work_dir)])
    assert result.exit_code == 0
    assert "Found 1 cuts" in result.stdout
    assert (work_dir / "cuts.json").exists()
    assert (work_dir / "cuts_report.md").exists()


def test_cli_render_dry_run_command(tmp_path: Path) -> None:
    from autocut.models import MediaInfo, SpeechInterval, SpeechIntervals, Transcript, TranscriptSegment, Word
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    media = MediaInfo(source=tmp_path / "video.mp4", duration=10.0)
    (work_dir / "ingest.json").write_text(media.model_dump_json(), encoding="utf-8")
    (work_dir / "audio.wav").touch()

    speech = SpeechIntervals(source="audio.wav", intervals=[SpeechInterval(start=0.0, end=2.0)])
    transcript = Transcript(
        source="audio.wav",
        model="test",
        language="ru",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="это типа важно",
                words=[
                    Word(word="это", start=0.0, end=0.2),
                    Word(word="типа", start=0.5, end=0.7),
                    Word(word="важно", start=0.75, end=1.0),
                ],
            )
        ],
    )
    (work_dir / "speech.json").write_text(speech.model_dump_json(), encoding="utf-8")
    (work_dir / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(app, ["render", str(work_dir), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry-run complete" in result.stdout
    assert "Original duration: 10.00s" in result.stdout
    assert (work_dir / "edl.json").exists()
    assert (work_dir / "subtitles.srt").exists()
    assert (work_dir / "subtitles.ass").exists()


def test_cli_subtitles_command(tmp_path: Path) -> None:
    from autocut.models import EditDecisionList, KeepRange, SourceMedia, Transcript, TranscriptSegment, Word
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    transcript = Transcript(
        source="audio.wav",
        model="test",
        language="ru",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="первое второе",
                words=[
                    Word(word="первое", start=0.2, end=0.8),
                    Word(word="второе", start=1.0, end=1.6),
                ],
            )
        ],
    )
    edl = EditDecisionList(
        source=SourceMedia(video=tmp_path / "video.mp4", duration=5.0),
        cuts=[],
        words=[],
        keep=[KeepRange(start=0.1, end=1.8)],
    )
    (work_dir / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    (work_dir / "edl.json").write_text(edl.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(app, ["subtitles", str(work_dir)])
    assert result.exit_code == 0
    assert "Generated subtitles" in result.stdout
    assert (work_dir / "subtitles.srt").exists()
    assert (work_dir / "subtitles.ass").exists()


def test_cli_analyze_with_markers(tmp_path: Path) -> None:
    import json
    from autocut.models import SpeechInterval, SpeechIntervals, Transcript, TranscriptSegment, Word
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    speech = SpeechIntervals(source="audio.wav", intervals=[SpeechInterval(start=0.0, end=5.0)])
    transcript = Transcript(
        source="audio.wav",
        model="test",
        language="ru",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=5.0,
                text="ошибка здесь и потом дубль два",
                words=[
                    Word(word="ошибка", start=1.0, end=1.5),
                    Word(word="здесь", start=1.6, end=2.0),
                    Word(word="и", start=2.2, end=2.4),
                    Word(word="потом", start=3.0, end=3.5),
                    Word(word="дубль", start=3.6, end=4.0),
                    Word(word="два", start=4.1, end=4.5),
                ],
            )
        ],
    )
    markers = [{"time": 2.5, "type": "bad_take"}]
    (work_dir / "markers.json").write_text(json.dumps(markers), encoding="utf-8")
    (work_dir / "speech.json").write_text(speech.model_dump_json(), encoding="utf-8")
    (work_dir / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(work_dir)])
    assert result.exit_code == 0
    cuts_data = json.loads((work_dir / "cuts.json").read_text(encoding="utf-8"))
    assert any(c.get("text") == "[bad_take marker]" for c in cuts_data)


