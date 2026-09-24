"""Probe source media and normalize its audio to 16 kHz mono WAV."""

import json
from pathlib import Path
import subprocess

from autocut.config import ensure_ffmpeg_in_path
from autocut.models import MediaInfo

ensure_ffmpeg_in_path()


def _parse_rate(rate_str: str) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    num, _, den = rate_str.partition("/")
    try:
        den_f = float(den) if den else 1.0
        return float(num) / den_f if den_f != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_media(source: Path) -> MediaInfo:
    """Read source duration and frame rate information with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-select_streams", "v:0", "-of", "json", str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration", 0.0))
    streams = payload.get("streams", [])
    is_vfr = False
    fps = None
    if streams:
        r_fps = _parse_rate(streams[0].get("r_frame_rate", "0/0"))
        avg_fps = _parse_rate(streams[0].get("avg_frame_rate", "0/0"))
        if r_fps > 0 and avg_fps > 0:
            is_vfr = abs(r_fps - avg_fps) > 0.05
        fps = round(r_fps if r_fps > 0 else (avg_fps if avg_fps > 0 else 30.0), 3)
    return MediaInfo(source=source, duration=duration, is_vfr=is_vfr, fps=fps)


def normalize_to_cfr(source: Path, output_path: Path, fps: float = 60.0) -> Path:
    """Normalize variable frame rate video to constant frame rate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-i", str(source),
        "-c:v", "h264_nvenc", "-r", str(fps),
        "-c:a", "copy", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        command[4] = "libx264"
        subprocess.run(command, check=True)
    return output_path


def extract_audio(source: Path, output_path: Path) -> Path:
    """Extract one PCM WAV suitable for bundled ONNX VAD and ASR."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )
    return output_path

