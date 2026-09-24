"""Probe source media and normalize its audio to 16 kHz mono WAV."""

import json
from pathlib import Path
import subprocess

from autocut.models import MediaInfo


def probe_media(source: Path) -> MediaInfo:
    """Read source duration with ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return MediaInfo(source=source, duration=float(payload["format"]["duration"]))


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
