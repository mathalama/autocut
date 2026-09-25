from pathlib import Path
from autocut.recorder import CaptionRecorder, format_srt_time


def test_format_srt_time():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.234) == "00:00:01,234"
    assert format_srt_time(65.5) == "00:01:05,500"
    assert format_srt_time(3661.05) == "01:01:01,050"


def test_caption_recorder(tmp_path: Path):
    srt_file = tmp_path / "test.srt"
    txt_file = tmp_path / "test.txt"

    recorder = CaptionRecorder(srt_path=srt_file, txt_path=txt_file)
    recorder.add_caption("Hello world", 1.0, 3.5)
    recorder.add_caption("Second line", 4.0, 6.2)

    srt_content = srt_file.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,500" in srt_content
    assert "Hello world" in srt_content
    assert "00:00:04,000 --> 00:00:06,200" in srt_content
    assert "Second line" in srt_content

    txt_content = txt_file.read_text(encoding="utf-8")
    assert txt_content.strip() == "Hello world\nSecond line"
