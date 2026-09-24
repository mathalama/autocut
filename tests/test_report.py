"""Stage 2 contract tests for Markdown cuts report."""

from pathlib import Path

from autocut.models import Cut, Word
from autocut.report import format_context, write_cut_report


def test_format_context_extracts_three_words_before_and_after() -> None:
    words = [
        Word(word="w1", start=0.0, end=0.1),
        Word(word="w2", start=0.1, end=0.2),
        Word(word="w3", start=0.2, end=0.3),
        Word(word="w4", start=0.3, end=0.4),
        Word(word="filler", start=0.5, end=0.7),
        Word(word="w5", start=0.8, end=0.9),
        Word(word="w6", start=0.9, end=1.0),
        Word(word="w7", start=1.0, end=1.1),
        Word(word="w8", start=1.1, end=1.2),
    ]
    cut = Cut(start=0.5, end=0.7, reason="filler", text="filler", confidence=0.7)
    ctx = format_context(cut, words)

    assert ctx == "w2 w3 w4 [filler] w5 w6 w7"


def test_write_cut_report_orders_by_confidence_ascending(tmp_path: Path) -> None:
    cuts = [
        Cut(start=2.0, end=2.5, reason="silence", confidence=0.95),
        Cut(start=0.5, end=0.8, reason="filler", text="типа", confidence=0.70),
        Cut(start=1.0, end=1.5, reason="repeat", text="мы сделаем", confidence=0.80),
    ]
    words = [
        Word(word="начало", start=0.0, end=0.4),
        Word(word="типа", start=0.5, end=0.8),
        Word(word="мы", start=1.0, end=1.2),
        Word(word="сделаем", start=1.2, end=1.5),
        Word(word="конец", start=2.6, end=3.0),
    ]
    report_file = tmp_path / "cuts_report.md"
    write_cut_report(cuts, words, report_file)

    content = report_file.read_text(encoding="utf-8")
    assert "# Cut report" in content
    assert "| Timecode | Reason | Text | Context | Confidence |" in content

    lines = [line for line in content.splitlines() if line.startswith("| 0.") or line.startswith("| 1.") or line.startswith("| 2.")]
    assert len(lines) == 3
    # Check ascending order of confidence: 0.70 first, then 0.80, then 0.95
    assert "0.70" in lines[0]
    assert "типа" in lines[0]
    assert "0.80" in lines[1]
    assert "мы сделаем" in lines[1]
    assert "0.95" in lines[2]
    assert "silence" in lines[2]


def test_write_cut_report_includes_pause_decisions(tmp_path: Path) -> None:
    from autocut.models import PauseDecision
    cuts = [Cut(start=2.0, end=2.5, reason="silence", confidence=0.95)]
    words = [Word(word="w1", start=0.0, end=0.5), Word(word="w2", start=3.0, end=3.5)]
    decisions = [
        PauseDecision(start=0.5, end=3.0, duration=2.5, decision="keep: input", details="Coding session protected: 12 events"),
        PauseDecision(start=5.0, end=10.0, duration=5.0, decision="compress: idle", details="Static idle screen"),
    ]
    report_file = tmp_path / "cuts_report.md"
    write_cut_report(cuts, words, report_file, pause_decisions=decisions)

    content = report_file.read_text(encoding="utf-8")
    assert "## Pause Decisions (Demonstration vs Idle)" in content
    assert "`keep: input`" in content
    assert "`compress: idle`" in content
    assert "Coding session protected: 12 events" in content


def test_generate_second_timeline(tmp_path: Path) -> None:
    from autocut.models import SpeechInterval, SpeechIntervals
    from autocut.report import generate_second_timeline

    cuts = [Cut(start=3.0, end=4.0, reason="silence", confidence=0.95)]
    speech = SpeechIntervals(source="audio.wav", intervals=[SpeechInterval(start=0.0, end=2.0)])
    input_events = [{"time": 1.2, "type": "key"}]
    timeline_file = tmp_path / "timeline_debug.txt"

    content = generate_second_timeline(
        duration=5.0,
        cuts=cuts,
        speech=speech,
        input_events=input_events,
        screen=None,
        output_path=timeline_file,
    )
    assert timeline_file.exists()
    assert "[00:01] VAD: SPEECH" in content
    assert "KEY(1)" in content
    assert "[00:03] VAD: SILENCE" in content
    assert "CUT (silence)" in content


