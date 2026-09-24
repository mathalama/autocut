"""Tests for numpy-based screen activity detection."""

from pathlib import Path
import subprocess

from autocut.config import ensure_ffmpeg_in_path
from autocut.stages.screen import detect_screen_activity

ensure_ffmpeg_in_path()


def test_detect_screen_activity_detects_motion(tmp_path: Path) -> None:
    video = tmp_path / "screen_test.mp4"
    out_json = tmp_path / "screen.json"

    # Create 3-second video: 0-1s static black, 1-2s animated testsrc (typing/motion), 2-3s static black
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=1",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x360:rate=10:duration=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=1",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        str(video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    activity = detect_screen_activity(video, output_path=out_json, fps=5)

    assert out_json.exists()
    assert len(activity.intervals) > 0
    # Motion occurred around 1.0 to 2.0s
    active_in_motion_window = any(
        (it.start <= 1.5 <= it.end) for it in activity.intervals
    )
    assert active_in_motion_window
