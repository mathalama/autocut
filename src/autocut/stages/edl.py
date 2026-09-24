"""Build a validated, non-overlapping edit decision list from cuts."""

from pathlib import Path

from autocut.config import EDL_MERGE_GAP_SECONDS, EDL_MIN_KEEP_SECONDS, EDL_WORD_PADDING_SECONDS
from autocut.models import Cut, EditDecisionList, KeepRange, SourceMedia, Word


def build_edl(
    video: Path | str,
    audio: Path | str | None,
    duration: float,
    cuts: list[Cut],
    words: list[Word],
) -> EditDecisionList:
    """Shrink cut edges by word padding, then complement them into keep ranges."""
    effective_cuts = sorted(
        (
            (max(0.0, cut.start + EDL_WORD_PADDING_SECONDS), min(duration, cut.end - EDL_WORD_PADDING_SECONDS))
            for cut in cuts
        ),
        key=lambda item: item[0],
    )
    merged_cuts: list[list[float]] = []
    for start, end in effective_cuts:
        if end <= start:
            continue
        if merged_cuts and start <= merged_cuts[-1][1]:
            merged_cuts[-1][1] = max(merged_cuts[-1][1], end)
        else:
            merged_cuts.append([start, end])

    keep: list[KeepRange] = []
    cursor = 0.0
    for start, end in merged_cuts:
        if start > cursor:
            keep.append(KeepRange(start=cursor, end=start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep.append(KeepRange(start=cursor, end=duration))

    merged_keep: list[KeepRange] = []
    for item in keep:
        if merged_keep and item.start - merged_keep[-1].end < EDL_MERGE_GAP_SECONDS:
            merged_keep[-1].end = item.end
        else:
            merged_keep.append(item)
    keep = [item for item in merged_keep if item.end - item.start >= EDL_MIN_KEEP_SECONDS]

    return EditDecisionList(
        source=SourceMedia(video=Path(video), audio=Path(audio) if audio else None, duration=duration),
        keep=keep,
        cuts=sorted(cuts, key=lambda cut: (cut.start, cut.end)),
        words=words,
    )
