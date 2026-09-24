"""Pydantic data contracts for the Stage 1 artifacts."""

from pathlib import Path

from pydantic import BaseModel, Field


class MediaInfo(BaseModel):
    """Duration discovered by ffprobe for an input media file."""

    source: Path
    duration: float = Field(ge=0)


class SpeechInterval(BaseModel):
    """A speech range on the original media timeline, in seconds."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)


class SpeechIntervals(BaseModel):
    """Serialized output of the VAD stage."""

    source: Path
    intervals: list[SpeechInterval]


class Word(BaseModel):
    """One recognized word and its original-timeline timestamps."""

    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)


class TranscriptSegment(BaseModel):
    """A recognized segment with word-level timing."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[Word]


class Transcript(BaseModel):
    """Serialized output of the ASR stage."""

    source: Path
    model: str
    language: str | None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment]
