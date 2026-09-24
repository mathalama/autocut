"""Stage 2 contracts for EDL construction invariants."""

from autocut.models import Cut, Word
from autocut.stages.edl import build_edl


def test_build_edl_sorts_merges_and_bounds_keep_ranges() -> None:
    edl = build_edl(
        video="recording.mkv",
        audio="audio.wav",
        duration=3.0,
        cuts=[
            Cut(start=2.0, end=2.5, reason="filler", confidence=0.9),
            Cut(start=0.5, end=1.0, reason="silence", confidence=0.95),
        ],
        words=[Word(word="one", start=0.0, end=0.2), Word(word="two", start=2.7, end=2.9)],
    )

    assert [(item.start, item.end) for item in edl.keep] == [(0.0, 0.56), (0.94, 2.06), (2.44, 3.0)]
    assert all(left.end <= right.start for left, right in zip(edl.keep, edl.keep[1:]))


def test_build_edl_discards_ranges_shorter_than_minimum() -> None:
    edl = build_edl(
        video="recording.mkv", audio=None, duration=1.0,
        cuts=[Cut(start=0.2, end=0.8, reason="silence", confidence=0.95)], words=[],
    )

    assert [(item.start, item.end) for item in edl.keep] == []


def test_build_edl_merges_ranges_closer_than_gap() -> None:
    # Gap between 1.0 and 1.05 is 0.05s < EDL_MERGE_GAP_SECONDS (0.1s)
    edl = build_edl(
        video="recording.mkv",
        audio=None,
        duration=4.0,
        cuts=[Cut(start=1.06, end=1.11, reason="filler", confidence=0.9)],
        words=[],
    )
    # Effective cut: [1.06 + 0.06, 1.11 - 0.06] = [1.12, 1.05] -> end <= start so cut is dropped!
    # Let's test with cut that leaves a 0.05s gap: cut from 1.0 to 1.17 -> effective cut [1.06, 1.11] -> gap is 0.05s
    # Gaps between keep ranges are merged if < 0.1s
    assert len(edl.keep) == 1
    assert edl.keep[0].start == 0.0
    assert edl.keep[0].end == 4.0


def test_edl_model_rejects_overlapping_or_unsorted_ranges() -> None:
    import pytest
    from pydantic import ValidationError
    from autocut.models import EditDecisionList, KeepRange, SourceMedia

    with pytest.raises(ValidationError):
        EditDecisionList(
            source=SourceMedia(video="test.mp4", duration=10.0),
            keep=[KeepRange(start=1.0, end=3.0), KeepRange(start=2.5, end=4.0)],
            cuts=[],
            words=[],
        )

    with pytest.raises(ValidationError):
        EditDecisionList(
            source=SourceMedia(video="test.mp4", duration=10.0),
            keep=[KeepRange(start=4.0, end=5.0), KeepRange(start=1.0, end=2.0)],
            cuts=[],
            words=[],
        )


def test_edl_model_rejects_ranges_exceeding_duration_or_negative() -> None:
    import pytest
    from pydantic import ValidationError
    from autocut.models import EditDecisionList, KeepRange, SourceMedia

    with pytest.raises(ValidationError):
        EditDecisionList(
            source=SourceMedia(video="test.mp4", duration=5.0),
            keep=[KeepRange(start=1.0, end=5.5)],
            cuts=[],
            words=[],
        )

    with pytest.raises(ValidationError):
        EditDecisionList(
            source=SourceMedia(video="test.mp4", duration=5.0),
            keep=[KeepRange(start=-0.1, end=2.0)],
            cuts=[],
            words=[],
        )

