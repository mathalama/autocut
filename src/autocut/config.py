"""Configuration defaults for real-time autocut."""

from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 800  # 50ms chunks at 16kHz

# English-specialized high-accuracy model (small.en)
DEFAULT_MODEL = "small.en"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "en"

# Timing settings (in seconds) - tuned for real-time single-pass subtitles
MIN_SPEECH_DURATION = 0.25      # Minimum speech duration to transcribe (ignore micro-clicks)
SILENCE_FINALIZE_SECONDS = 0.40  # Natural speech pause threshold to finalize subtitle
MAX_SPEECH_DURATION = 3.80      # Max utterance duration before emitting subtitle during continuous speech
FADE_OUT_SECONDS = 3.0          # Time of silence before subtitles fade out on screen

# Server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

import sys
if getattr(sys, "frozen", False):
    base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    candidate = base_dir / "autocut" / "static"
    STATIC_DIR = candidate if candidate.exists() else (base_dir / "static")
else:
    STATIC_DIR = Path(__file__).resolve().parent / "static"
