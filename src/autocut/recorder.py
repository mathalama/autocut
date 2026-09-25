"""Recorder for live captions to SRT and TXT formats."""

from datetime import timedelta
from pathlib import Path
from typing import Optional


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    td = timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class CaptionRecorder:
    """Records finalized captions into .srt and .txt files."""

    def __init__(self, srt_path: Optional[Path] = None, txt_path: Optional[Path] = None) -> None:
        self.srt_path = srt_path
        self.txt_path = txt_path
        self.counter = 1
        if self.srt_path:
            self.srt_path.parent.mkdir(parents=True, exist_ok=True)
            self.srt_path.write_text("", encoding="utf-8")
        if self.txt_path:
            self.txt_path.parent.mkdir(parents=True, exist_ok=True)
            self.txt_path.write_text("", encoding="utf-8")

    def add_caption(self, text: str, start_time: float, end_time: float) -> None:
        """Add a finalized caption entry."""
        clean_text = text.strip()
        if not clean_text:
            return

        if self.srt_path:
            srt_entry = (
                f"{self.counter}\n"
                f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
                f"{clean_text}\n\n"
            )
            with open(self.srt_path, "a", encoding="utf-8") as f:
                f.write(srt_entry)
            self.counter += 1

        if self.txt_path:
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write(f"{clean_text}\n")
