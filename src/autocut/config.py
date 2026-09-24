import os
from pathlib import Path
import shutil

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_INITIAL_PROMPT = "Э-э, эм, эээ, ну, типа, как бы, вот, короче. Um, uh, like, you know."
SAMPLE_RATE = 16_000

# Values are passed to faster-whisper's bundled ONNX Silero VAD.
VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 80,
}

PAUSE_THRESHOLD_SECONDS = 0.7
PAUSE_RETAIN_SECONDS = 0.25
IDLE_CUT_THRESHOLD_SECONDS = 3.5  # Dead-zone threshold: gaps <= 3.5s within action sessions are NEVER cut
CONTEXTUAL_FILLER_PAUSE_SECONDS = 0.15
INPUT_ACTION_PADDING_SECONDS = 2.0  # Window p around user actions (g <= 2p merges actions)
SCREEN_DIFF_THRESHOLD = 25
SCREEN_MIN_CHANGED_FRACTION = 0.001  # 0.1% of frame area (replaces fixed pixel counts across resolutions)
DEFAULT_HOTKEY = "f12"
DEFAULT_SYNC_KEY = "scroll_lock"
EDL_WORD_PADDING_SECONDS = 0.06
EDL_MERGE_GAP_SECONDS = 0.1
EDL_MIN_KEEP_SECONDS = 0.3
RENDER_ACROSSFADE_SECONDS = 0.02

# Candidate directories for filler data files
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FILLER_FILES = (
    _REPO_ROOT / "data" / "fillers_ru.txt",
    _REPO_ROOT / "data" / "fillers_en.txt",
)

CUT_CONFIDENCE = {"silence": 0.95, "filler_always": 0.9, "filler_contextual": 0.7, "repeat": 0.8}

GYAN_FFMPEG_BIN = Path(
    r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin"
)


def ensure_ffmpeg_in_path() -> None:
    """Ensure ffmpeg and ffprobe resolve in environment PATH."""
    if shutil.which("ffmpeg") is None and GYAN_FFMPEG_BIN.exists():
        os.environ["PATH"] = str(GYAN_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")


ensure_ffmpeg_in_path()


def load_fillers() -> dict[str, set[tuple[str, ...]]]:
    """Load ``always`` and ``contextual`` phrases from tracked data files."""
    groups: dict[str, set[tuple[str, ...]]] = {"always": set(), "contextual": set()}
    for path in FILLER_FILES:
        if not path.exists():
            fallback = Path("data") / path.name
            if fallback.exists():
                path = fallback
            else:
                continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            group, phrase = line.split("\t", maxsplit=1)
            groups[group].add(tuple(phrase.casefold().split()))
    return groups

