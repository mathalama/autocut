"""Unit tests for Stage 3 timestamp-remapping and cue grouping."""

from autocut.models import KeepRange, Word
from autocut.stages.remap import group_words_into_cues, remap_words


def test_remap_words_shifts_timestamps_according_to_keep_ranges() -> None:
    # 2 keep ranges: [1.0, 3.0] (len 2.0s) and [5.0, 7.0] (len 2.0s)
    # Total edited length: 4.0s
    keep = [KeepRange(start=1.0, end=3.0), KeepRange(start=5.0, end=7.0)]

    words = [
        Word(word="w1", start=1.2, end=1.8),  # In range 1: shifted by -1.0 -> [0.2, 0.8]
        Word(word="cut", start=3.5, end=4.5), # In cut interval: dropped!
        Word(word="w2", start=5.2, end=5.8),  # In range 2: offset is 2.0, shifted by -5.0 + 2.0 -> [2.2, 2.8]
    ]

    remapped = remap_words(words, keep)

    assert len(remapped) == 2
    assert remapped[0].word == "w1"
    assert remapped[0].start == 0.2
    assert remapped[0].end == 0.8

    assert remapped[1].word == "w2"
    assert remapped[1].start == 2.2
    assert remapped[1].end == 2.8


def test_remap_words_empty_or_no_keep() -> None:
    words = [Word(word="test", start=1.0, end=2.0)]
    assert remap_words(words, []) == []
    assert remap_words([], [KeepRange(start=0.0, end=5.0)]) == []


def test_group_words_into_cues() -> None:
    words = [
        Word(word="Привет,", start=0.1, end=0.4),
        Word(word="это", start=0.5, end=0.7),
        Word(word="первая", start=0.75, end=1.0),
        Word(word="фраза.", start=1.05, end=1.4),
        Word(word="А", start=2.5, end=2.7),  # Gap 1.1s > max_gap (0.6s) -> new cue
        Word(word="это", start=2.75, end=3.0),
        Word(word="вторая.", start=3.05, end=3.5),
    ]

    cues = group_words_into_cues(words, max_words=6, max_gap=0.6)

    assert len(cues) == 2
    assert cues[0][2] == "Привет, это первая фраза."
    assert cues[0][0] == 0.1
    assert cues[0][1] == 1.4

    assert cues[1][2] == "А это вторая."
    assert cues[1][0] == 2.5
    assert cues[1][1] == 3.5
