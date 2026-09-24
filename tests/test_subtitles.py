"""Unit tests for subtitle file creation (.srt and .ass)."""

from pathlib import Path
from autocut.models import Word
from autocut.stages.subtitles import generate_subtitles


def test_generate_subtitles_writes_valid_srt_and_ass(tmp_path: Path) -> None:
    words = [
        Word(word="Привет", start=0.5, end=0.9),
        Word(word="мир!", start=1.0, end=1.5),
    ]
    srt_path = tmp_path / "subtitles.srt"
    ass_path = tmp_path / "subtitles.ass"

    out_srt, out_ass = generate_subtitles(words, srt_path, ass_path)

    assert out_srt.exists()
    assert out_ass.exists()

    srt_content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:01,500" in srt_content
    assert "Привет мир!" in srt_content

    ass_content = ass_path.read_text(encoding="utf-8")
    assert "[Script Info]" in ass_content
    assert "[V4+ Styles]" in ass_content
    assert "Default,Segoe UI" in ass_content
    assert "Dialogue:" in ass_content
    assert r"{\k" in ass_content  # Karaoke timing tags
