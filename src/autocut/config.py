"""Configuration defaults for the approved Stage 1 pipeline."""

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
SAMPLE_RATE = 16_000

# Values are passed to faster-whisper's bundled ONNX Silero VAD.
VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 80,
}
