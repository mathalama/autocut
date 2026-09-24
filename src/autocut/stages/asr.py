"""Transcribe approved VAD speech ranges with word-level timestamps."""

from pathlib import Path

from faster_whisper import WhisperModel

from autocut.models import SpeechInterval, Transcript, TranscriptSegment, Word


def transcribe(
    audio_path: Path,
    intervals: list[SpeechInterval],
    output_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    initial_prompt: str | None,
) -> Transcript:
    """Transcribe only supplied speech intervals and persist a JSON artifact."""
    if not intervals:
        raise ValueError("No speech intervals found; refusing to transcribe silence.")

    model = WhisperModel(
        model_size_or_path=model_name,
        device=device,
        compute_type=compute_type,
    )
    clip_timestamps = [point for interval in intervals for point in (interval.start, interval.end)]
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        vad_filter=False,
        clip_timestamps=clip_timestamps,
    )
    transcript = Transcript(
        source=audio_path,
        model=model_name,
        language=info.language,
        language_probability=info.language_probability,
        segments=[
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=[
                    Word(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        probability=word.probability,
                    )
                    for word in (segment.words or [])
                ],
            )
            for segment in segments
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript
