"""Use faster-whisper's bundled ONNX Silero VAD to find speech ranges."""

from pathlib import Path

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from autocut.config import SAMPLE_RATE, VAD_PARAMETERS
from autocut.models import SpeechInterval, SpeechIntervals


def detect_speech(audio_path: Path, output_path: Path) -> SpeechIntervals:
    """Write speech intervals in seconds for a normalized 16 kHz WAV."""
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    timestamps = get_speech_timestamps(audio, VadOptions(**VAD_PARAMETERS))
    result = SpeechIntervals(
        source=audio_path,
        intervals=[
            SpeechInterval(start=item["start"] / SAMPLE_RATE, end=item["end"] / SAMPLE_RATE)
            for item in timestamps
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
