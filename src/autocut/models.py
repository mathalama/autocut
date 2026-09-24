"""Pydantic data contracts for the Stage 1 artifacts."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MediaInfo(BaseModel):
    """Duration and frame rate discovered by ffprobe for an input media file."""

    source: Path
    duration: float = Field(ge=0)
    is_vfr: bool = False
    fps: float | None = None


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


class Cut(BaseModel):
    """A source-timeline interval to remove, with its rationale."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    reason: Literal["silence", "filler", "repeat"]
    text: str | None = None
    confidence: float = Field(ge=0, le=1)


class KeepRange(BaseModel):
    """A non-overlapping source interval retained by the edit decision list."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)


class SourceMedia(BaseModel):
    """Source paths and duration represented by an EDL."""

    video: Path
    audio: Path | None = None
    duration: float = Field(ge=0)


class EditDecisionList(BaseModel):
    """Non-destructive edit list consumed by future pipeline stages."""

    version: Literal[1] = 1
    source: SourceMedia
    keep: list[KeepRange]
    cuts: list[Cut]
    words: list[Word]

    @model_validator(mode="after")
    def validate_invariants(self) -> "EditDecisionList":
        # 1. keep bounds within source duration
        for item in self.keep:
            if item.start < 0:
                raise ValueError(f"Keep range start {item.start} is negative.")
            if item.end > self.source.duration + 1e-4:
                raise ValueError(
                    f"Keep range end {item.end} exceeds source duration {self.source.duration}."
                )
            if item.start > item.end:
                raise ValueError(f"Keep range start {item.start} exceeds end {item.end}.")

        # 2. keep is sorted and non-overlapping
        for prev, curr in zip(self.keep, self.keep[1:]):
            if curr.start < prev.end:
                raise ValueError(
                    f"Keep ranges overlap or are not sorted: ({prev.start}, {prev.end}) and ({curr.start}, {curr.end})."
                )
        return self


class ScreenActivityInterval(BaseModel):
    """Interval where screen was active (e.g. typing, scrolling, compilation)."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    changed_pixels: int = 0
    activity_type: Literal["typing", "motion", "macro"] = "typing"


class ScreenActivity(BaseModel):
    """Serialized output of the screen motion detection stage."""

    source: Path
    intervals: list[ScreenActivityInterval]


class InputEvent(BaseModel):
    """Anonymous input event timestamp (key, click, scroll, window, bad_take)."""

    time: float = Field(ge=0)
    type: Literal["key", "click", "scroll", "window", "bad_take"]
    window_title: str | None = None


class PauseDecision(BaseModel):
    """Evaluation of a silence interval between speech words."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    duration: float = Field(ge=0)
    decision: str  # e.g. "keep: input", "keep: screen", "compress: idle"
    details: str = ""


