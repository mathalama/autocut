r"""Extract random timestamped-word clips for a human timing check.

Example:
  .\.venv\Scripts\python.exe scripts\check_word_bounds.py \
    --transcript work\recording\transcript.json --audio work\recording\audio.wav
"""

import argparse
import json
from pathlib import Path
import random
import re
import subprocess

PADDING_SECONDS = 0.060


def parse_args() -> argparse.Namespace:
    """Parse explicit artifact paths so the script has no hidden state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def words_from_transcript(transcript: dict) -> list[dict]:
    """Flatten valid word timestamps while retaining their original text."""
    return [
        word
        for segment in transcript.get("segments", [])
        for word in segment.get("words", [])
        if word.get("word", "").strip() and word.get("end", 0) > word.get("start", 0)
    ]


def safe_name(word: str) -> str:
    """Produce a portable filename component from a transcript token."""
    return re.sub(r"[^\w.-]+", "_", word.strip(), flags=re.UNICODE).strip("_.") or "word"


def extract_clip(audio: Path, output: Path, start: float, end: float) -> None:
    """Extract a lossless WAV clip in the source's original timeline."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(audio),
            "-t", f"{end - start:.3f}", "-vn", "-c:a", "pcm_s16le", str(output),
        ],
        check=True,
    )


def main() -> None:
    """Sample words, create listening clips, and write their manifest."""
    args = parse_args()
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    words = words_from_transcript(transcript)
    if not words:
        raise SystemExit("No timestamped words found in transcript.")
    if args.count < 1:
        raise SystemExit("--count must be positive.")

    output_dir = args.output_dir or args.transcript.parent / "word_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = random.Random(args.seed).sample(words, min(args.count, len(words)))
    manifest = []
    for index, word in enumerate(selected, start=1):
        start = max(0.0, float(word["start"]) - PADDING_SECONDS)
        end = float(word["end"]) + PADDING_SECONDS
        clip_name = f"{index:02d}_{start:.3f}-{end:.3f}_{safe_name(word['word'])}.wav"
        extract_clip(args.audio, output_dir / clip_name, start, end)
        manifest.append({"file": clip_name, "word": word["word"], "start": start, "end": end})

    (output_dir / "manifest.json").write_text(
        json.dumps({"padding_seconds": PADDING_SECONDS, "clips": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} clips to {output_dir}")


if __name__ == "__main__":
    main()
