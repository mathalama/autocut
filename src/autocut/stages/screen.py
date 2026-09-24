"""Detect screen activity (typing, terminal scroll, macro motion) via numpy frame diffs."""

from pathlib import Path
import subprocess

import numpy as np

from autocut.config import SCREEN_DIFF_THRESHOLD, SCREEN_MIN_CHANGED_FRACTION, ensure_ffmpeg_in_path
from autocut.models import ScreenActivity, ScreenActivityInterval

ensure_ffmpeg_in_path()


def detect_screen_activity(
    video_path: Path,
    output_path: Path | None = None,
    fps: int = 5,
    width: int = 640,
    height: int = 360,
    diff_threshold: int = SCREEN_DIFF_THRESHOLD,
    min_changed_pixels: int | None = None,
    min_changed_fraction: float = SCREEN_MIN_CHANGED_FRACTION,
    merge_gap_seconds: float = 0.8,
) -> ScreenActivity:
    """Read 640x360 gray frames at target fps and detect localized typing and motion intervals."""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height
    frame_duration = 1.0 / fps

    if min_changed_pixels is None:
        effective_min_pixels = max(10, int(frame_size * min_changed_fraction))
    else:
        effective_min_pixels = min_changed_pixels

    macro_threshold_pixels = max(100, int(frame_size * 0.003))  # 0.3% of frame

    active_spans: list[tuple[float, float, int, str]] = []
    prev_frame: np.ndarray | None = None
    frame_idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        curr_frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        if prev_frame is not None:
            diff = np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16))
            diff_mask = diff > diff_threshold
            changed_pixels = int(np.sum(diff_mask))

            # Terminal builds, docker pull, npm install and progress bars change
            # characters or bars across frames. Bounded cursor blink (<10px) is rejected.
            if changed_pixels >= effective_min_pixels:
                t_start = round((frame_idx - 1) * frame_duration, 3)
                t_end = round(frame_idx * frame_duration, 3)
                act_type = "motion" if changed_pixels >= macro_threshold_pixels else "typing"
                active_spans.append((t_start, t_end, changed_pixels, act_type))

        prev_frame = curr_frame
        frame_idx += 1

    proc.wait()

    # Merge nearby spans (e.g. intervals separated by brief typing pauses < merge_gap_seconds)
    merged_intervals: list[ScreenActivityInterval] = []
    if active_spans:
        cur_start, cur_end, cur_pixels, cur_type = active_spans[0]
        for s_start, s_end, s_pixels, s_type in active_spans[1:]:
            if s_start - cur_end <= merge_gap_seconds:
                cur_end = max(cur_end, s_end)
                cur_pixels = max(cur_pixels, s_pixels)
                if s_type == "motion":
                    cur_type = "motion"
            else:
                merged_intervals.append(
                    ScreenActivityInterval(
                        start=round(cur_start, 3),
                        end=round(cur_end, 3),
                        changed_pixels=cur_pixels,
                        activity_type=cur_type,
                    )
                )
                cur_start, cur_end, cur_pixels, cur_type = s_start, s_end, s_pixels, s_type

        merged_intervals.append(
            ScreenActivityInterval(
                start=round(cur_start, 3),
                end=round(cur_end, 3),
                changed_pixels=cur_pixels,
                activity_type=cur_type,
            )
        )

    result = ScreenActivity(source=video_path, intervals=merged_intervals)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
