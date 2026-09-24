"""Stage 2 contracts for pause, filler, and repeat detection."""

from autocut.models import SpeechInterval, SpeechIntervals, Transcript, TranscriptSegment, Word
from autocut.stages.analyze import analyze


def transcript(words: list[tuple[str, float, float]]) -> Transcript:
    return Transcript(
        source="audio.wav", model="test", language="ru", segments=[TranscriptSegment(
            start=words[0][1], end=words[-1][2], text=" ".join(word for word, _, _ in words),
            words=[Word(word=word, start=start, end=end) for word, start, end in words],
        )],
    )


def test_analyze_compresses_only_the_middle_of_long_pause() -> None:
    result = analyze(transcript([("один", 0.0, 0.2), ("два", 1.2, 1.4)]), SpeechIntervals(source="audio.wav", intervals=[]))

    pauses = [(cut.start, cut.end) for cut in result if cut.reason == "silence"]
    assert pauses == [(0.325, 1.075)]


def test_analyze_keeps_contextual_filler_inside_fluid_phrase() -> None:
    result = analyze(transcript([("это", 0.0, 0.2), ("типа", 0.25, 0.45), ("важно", 0.5, 0.8)]), SpeechIntervals(source="audio.wav", intervals=[]))

    assert not any(cut.reason == "filler" for cut in result)


def test_analyze_removes_contextual_filler_after_a_pause() -> None:
    result = analyze(transcript([("это", 0.0, 0.2), ("типа", 0.5, 0.7), ("важно", 0.75, 1.0)]), SpeechIntervals(source="audio.wav", intervals=[]))

    assert [(cut.text, cut.reason) for cut in result if cut.reason == "filler"] == [("типа", "filler")]


def test_analyze_removes_early_repeated_phrase_but_not_one_letter_words() -> None:
    result = analyze(transcript([("мы", 0.0, 0.2), ("сделаем", 0.2, 0.5), ("мы", 0.5, 0.7), ("сделаем", 0.7, 1.0), ("и", 1.0, 1.1), ("и", 1.1, 1.2)]), SpeechIntervals(source="audio.wav", intervals=[]))

    repeats = [cut for cut in result if cut.reason == "repeat"]
    assert [(cut.text, cut.start, cut.end) for cut in repeats] == [("мы сделаем", 0.0, 0.5)]


def test_analyze_always_filler_removed_in_fluid_phrase() -> None:
    result = analyze(transcript([("это", 0.0, 0.2), ("эм", 0.22, 0.4), ("важно", 0.42, 0.7)]), SpeechIntervals(source="audio.wav", intervals=[]))

    fillers = [cut for cut in result if cut.reason == "filler"]
    assert len(fillers) == 1
    assert fillers[0].text == "эм"
    assert fillers[0].confidence == 0.9


def test_analyze_ignores_short_pause() -> None:
    result = analyze(transcript([("слово", 0.0, 0.5), ("другое", 0.8, 1.2)]), SpeechIntervals(source="audio.wav", intervals=[]))

    assert not any(cut.reason == "silence" for cut in result)


def test_analyze_repeats_up_to_four_words() -> None:
    words = [
        ("первый", 0.0, 0.2), ("второй", 0.2, 0.4), ("третий", 0.4, 0.6), ("четвертый", 0.6, 0.8),
        ("первый", 0.8, 1.0), ("второй", 1.0, 1.2), ("третий", 1.2, 1.4), ("четвертый", 1.4, 1.6),
    ]
    result = analyze(transcript(words), SpeechIntervals(source="audio.wav", intervals=[]))

    repeats = [cut for cut in result if cut.reason == "repeat"]
    assert len(repeats) == 1
    assert repeats[0].text == "первый второй третий четвертый"
    assert repeats[0].start == 0.0
    assert repeats[0].end == 0.8


def test_analyze_repeats_ignores_digits_and_number_words() -> None:
    result = analyze(transcript([("42", 0.0, 0.3), ("42", 0.35, 0.6), ("два", 0.65, 0.9), ("два", 0.95, 1.2)]), SpeechIntervals(source="audio.wav", intervals=[]))

    repeats = [cut for cut in result if cut.reason == "repeat"]
    assert repeats == []


def test_analyze_empty_transcript() -> None:
    empty_trans = Transcript(source="audio.wav", model="test", language="ru", segments=[])
    result = analyze(empty_trans, SpeechIntervals(source="audio.wav", intervals=[]))

    assert result == []


def test_analyze_repeats_ignores_reduplication_exclusions() -> None:
    words = [
        ("да", 0.0, 0.2), ("да", 0.25, 0.45),
        ("очень", 0.5, 0.8), ("очень", 0.85, 1.15),
        ("тест", 1.2, 1.4), ("тест", 1.45, 1.65),
    ]
    result = analyze(transcript(words), SpeechIntervals(source="audio.wav", intervals=[]))
    repeats = [cut for cut in result if cut.reason == "repeat"]
    assert repeats == []


def test_analyze_clamps_silence_cut_against_vad_speech() -> None:
    # Gap between words is 0.2 to 1.2 (unclamped cut would be 0.325 to 1.075)
    # But VAD detects acoustic sound / breath up to 0.45!
    speech = SpeechIntervals(
        source="audio.wav",
        intervals=[SpeechInterval(start=0.0, end=0.45), SpeechInterval(start=1.2, end=1.4)],
    )
    result = analyze(transcript([("один", 0.0, 0.2), ("два", 1.2, 1.4)]), speech)
    pauses = [cut for cut in result if cut.reason == "silence"]
    assert pauses[0].start == 0.47
    assert pauses[0].end == 1.075


def test_analyze_preserves_silent_pause_when_user_is_typing() -> None:
    # 5-second silence between words: user was typing code silently
    words = [("начало", 0.0, 0.5), ("конец", 5.5, 6.0)]
    input_events = [
        {"time": 1.2, "type": "key"},
        {"time": 2.5, "type": "key"},
        {"time": 3.8, "type": "click"},
    ]
    result = analyze(
        transcript(words),
        SpeechIntervals(source="audio.wav", intervals=[]),
        markers=input_events,
    )
    # The pause must NOT be cut because the user was demonstrating/typing!
    assert not any(c.reason == "silence" for c in result)


def test_analyze_preserves_silent_pause_when_screen_is_active() -> None:
    from autocut.models import ScreenActivity, ScreenActivityInterval
    words = [("начало", 0.0, 0.5), ("конец", 5.5, 6.0)]
    # Screen had terminal output / code compilation between 1.0 and 4.0
    screen = ScreenActivity(
        source="video.mp4",
        intervals=[ScreenActivityInterval(start=1.0, end=4.0, changed_pixels=800, activity_type="typing")],
    )
    result = analyze(
        transcript(words),
        SpeechIntervals(source="audio.wav", intervals=[]),
        screen=screen,
    )
    # The pause must NOT be cut because the screen was actively changing!
    assert not any(c.reason == "silence" for c in result)


def test_analyze_partial_idle_gap_cuts_only_dead_zone() -> None:
    # 19s pause: user types at t=3.0 and at t=18.0.
    # Protected windows (p=2.0): [1.0, 5.0] and [16.0, 20.0].
    # Dead zone: [5.0, 16.0] (11.0s). Margin = 0.125s.
    # Expected cut: [5.125, 15.875].
    words = [("слово1", 0.0, 1.0), ("слово2", 20.0, 21.0)]
    input_events = [
        {"time": 3.0, "type": "key"},
        {"time": 18.0, "type": "key"},
    ]
    cuts, decisions = analyze(
        transcript(words),
        SpeechIntervals(source="audio.wav", intervals=[]),
        markers=input_events,
        return_decisions=True,
    )
    silence_cuts = [c for c in cuts if c.reason == "silence"]
    assert len(silence_cuts) == 1
    assert abs(silence_cuts[0].start - 5.125) < 1e-4
    assert abs(silence_cuts[0].end - 15.875) < 1e-4

    assert len(decisions) == 1
    assert decisions[0].decision == "compress: idle"
    assert "Partial idle" in decisions[0].details


def test_analyze_session_merges_when_gap_leq_2p() -> None:
    # 11s pause: user types at 3.0, 6.5, 9.5 (gaps 3.5s and 3.0s <= 2p = 4.0s).
    # All windows merge into continuous coding session [1.0, 11.5].
    words = [("начало", 0.0, 1.0), ("конец", 12.0, 13.0)]
    input_events = [
        {"time": 3.0, "type": "key"},
        {"time": 6.5, "type": "key"},
        {"time": 9.5, "type": "key"},
    ]
    cuts, decisions = analyze(
        transcript(words),
        SpeechIntervals(source="audio.wav", intervals=[]),
        markers=input_events,
        return_decisions=True,
    )
    silence_cuts = [c for c in cuts if c.reason == "silence"]
    assert len(silence_cuts) == 0
    assert len(decisions) == 1
    assert decisions[0].decision == "keep: input"
    assert "Coding session protected" in decisions[0].details




