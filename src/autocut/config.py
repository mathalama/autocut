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

# Timing settings (in seconds) - tuned for real-time low latency
STEP_SECONDS = 0.15         # Run inference every 150ms when speech is active
MIN_SPEECH_DURATION = 0.20  # Minimum speech duration to transcribe
SILENCE_FINALIZE_SECONDS = 0.6  # Pause duration to mark phrase as final
FADE_OUT_SECONDS = 2.5       # Time of silence before subtitles fade out on screen

# Server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_WS_PORT = 8766

STATIC_DIR = Path(__file__).resolve().parent / "static"
