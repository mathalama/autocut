"""Detect conservative Stage 2 cuts from words on the source timeline."""

import re

from autocut.config import (
    CONTEXTUAL_FILLER_PAUSE_SECONDS,
    CUT_CONFIDENCE,
    IDLE_CUT_THRESHOLD_SECONDS,
    INPUT_ACTION_PADDING_SECONDS,
    PAUSE_RETAIN_SECONDS,
    PAUSE_THRESHOLD_SECONDS,
    load_fillers,
)
from autocut.models import Cut, PauseDecision, ScreenActivity, SpeechIntervals, Transcript, Word


def _normalized(word: Word) -> str:
    return re.sub(r"[^\w]+", "", word.word.casefold(), flags=re.UNICODE)


def _flatten(transcript: Transcript) -> list[Word]:
    return [word for segment in transcript.segments for word in segment.words]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        last_s, last_e = merged[-1]
        if start <= last_e:
            merged[-1] = (last_s, max(last_e, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    base: tuple[float, float], subtract: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Subtract overlapping intervals from a base (start, end) interval, returning remaining gaps."""
    b_start, b_end = base
    if b_end <= b_start:
        return []
    merged_sub = _merge_intervals(subtract)
    idle: list[tuple[float, float]] = []
    curr = b_start
    for s_start, s_end in merged_sub:
        c_start = max(b_start, s_start)
        c_end = min(b_end, s_end)
        if c_end <= c_start:
            continue
        if c_start > curr:
            idle.append((curr, c_start))
        curr = max(curr, c_end)
    if curr < b_end:
        idle.append((curr, b_end))
    return idle


def is_silent_demonstration(
    start: float,
    end: float,
    input_events: list[dict] | None = None,
    screen: ScreenActivity | None = None,
) -> bool:
    """Check if silence interval contains active keyboard typing, clicks, or screen changes."""
    # 1. Check keyboard/mouse input during this silence:
    if input_events:
        for event in input_events:
            t = float(event.get("time", -1.0))
            if start <= t <= end and event.get("type") in ("key", "click", "scroll"):
                return True

    # 2. Check screen activity (tile/pixel diffs):
    if screen and screen.intervals:
        for interval in screen.intervals:
            overlap_start = max(start, interval.start)
            overlap_end = min(end, interval.end)
            if overlap_end - overlap_start >= 0.4:  # At least 400ms of screen activity
                return True

    return False


def _evaluate_pauses(
    words: list[Word],
    input_events: list[dict] | None = None,
    screen: ScreenActivity | None = None,
    p: float = INPUT_ACTION_PADDING_SECONDS,
    pause_threshold: float = PAUSE_THRESHOLD_SECONDS,
    idle_cut_threshold: float = IDLE_CUT_THRESHOLD_SECONDS,
    pause_retain: float = PAUSE_RETAIN_SECONDS,
) -> tuple[list[Cut], list[PauseDecision]]:
    cuts: list[Cut] = []
    decisions: list[PauseDecision] = []
    margin = pause_retain / 2.0

    for previous, current in zip(words, words[1:]):
        gap_start = previous.end
        gap_end = current.start
        gap = gap_end - gap_start
        if gap <= pause_threshold:
            continue

        # 1. Collect protected spans from input actions
        # Each event at time t protects [t - p, t + p]. If two events are <= 2p apart, they merge.
        input_spans: list[tuple[float, float]] = []
        matching_events: list[dict] = []
        if input_events:
            for ev in input_events:
                t = float(ev.get("time", -1.0))
                ev_type = ev.get("type", "")
                if ev_type in ("key", "click", "scroll") and (gap_start - p <= t <= gap_end + p):
                    input_spans.append((max(gap_start, t - p), min(gap_end, t + p)))
                    matching_events.append(ev)

        # 2. Collect protected spans from screen activity (includes tail p to read results)
        screen_spans: list[tuple[float, float]] = []
        if screen and screen.intervals:
            for s_int in screen.intervals:
                overlap_s = max(gap_start, s_int.start)
                overlap_e = min(gap_end, s_int.end + p)
                if overlap_e - overlap_s >= 0.3:
                    screen_spans.append((overlap_s, overlap_e))

        all_protected = _merge_intervals(input_spans + screen_spans)
        idle_spans = _subtract_intervals((gap_start, gap_end), all_protected)

        # Evaluate idle spans to determine cuts:
        # If user actions or screen changes were present in this pause, use idle_cut_threshold (3.5s)
        # to NEVER cut 1-2s pauses within sessions.
        # If no actions were present at all, use pause_threshold (0.7s) to compress conversational dead air.
        effective_threshold = idle_cut_threshold if (input_spans or screen_spans) else pause_threshold

        pause_cuts: list[Cut] = []
        for idle_s, idle_e in idle_spans:
            idle_len = idle_e - idle_s
            if idle_len > effective_threshold:
                cut_s = idle_s + margin
                cut_e = idle_e - margin
                if cut_e > cut_s:
                    pause_cuts.append(
                        Cut(
                            start=round(cut_s, 6),
                            end=round(cut_e, 6),
                            reason="silence",
                            text=None,
                            confidence=CUT_CONFIDENCE["silence"],
                        )
                    )

        cuts.extend(pause_cuts)

        # Determine report decision
        if not pause_cuts:
            if input_spans:
                dec = "keep: input"
                det = f"Coding session protected: {len(matching_events)} events merged with +/-{p}s window"
            elif screen_spans:
                dec = "keep: screen"
                det = "Screen motion/demo active throughout pause"
            else:
                dec = "natural"
                det = f"Short gap {gap:.2f}s kept naturally"
        else:
            if not input_spans and not screen_spans:
                dec = "compress: idle"
                det = f"Static idle screen ({gap:.2f}s -> {pause_retain:.2f}s)"
            else:
                dec = "compress: idle"
                total_cut = sum(c.end - c.start for c in pause_cuts)
                det = f"Partial idle ({total_cut:.2f}s cut, active coding/screen demonstration preserved)"

        decisions.append(
            PauseDecision(
                start=round(gap_start, 3),
                end=round(gap_end, 3),
                duration=round(gap, 3),
                decision=dec,
                details=det,
            )
        )

    return cuts, decisions


def _pause_cuts(
    words: list[Word],
    input_events: list[dict] | None = None,
    screen: ScreenActivity | None = None,
) -> list[Cut]:
    cuts, _ = _evaluate_pauses(words, input_events, screen)
    return cuts


def _filler_cuts(words: list[Word]) -> list[Cut]:
    fillers = load_fillers()
    normalized = [_normalized(word) for word in words]
    cuts: list[Cut] = []
    for group, confidence_key in (("always", "filler_always"), ("contextual", "filler_contextual")):
        for phrase in fillers[group]:
            size = len(phrase)
            for index in range(len(words) - size + 1):
                if tuple(normalized[index:index + size]) != phrase:
                    continue
                before_gap = float("inf") if index == 0 else words[index].start - words[index - 1].end
                after_gap = float("inf") if index + size == len(words) else words[index + size].start - words[index + size - 1].end
                contextual_ok = before_gap > CONTEXTUAL_FILLER_PAUSE_SECONDS or after_gap > CONTEXTUAL_FILLER_PAUSE_SECONDS
                if group == "always" or contextual_ok:
                    cuts.append(Cut(
                        start=words[index].start,
                        end=words[index + size - 1].end,
                        reason="filler",
                        text=" ".join(word.word.strip() for word in words[index:index + size]),
                        confidence=CUT_CONFIDENCE[confidence_key],
                    ))
    return cuts


NUMBER_WORDS = {
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}

REDUPLICATION_EXCLUSIONS = {
    "да", "нет", "очень", "так", "еще", "ещё", "чуть", "раз", "быстро", "едва", "прям", "прямо", "тоже",
    "yes", "no", "very", "so", "really", "now", "well", "test", "тест",
}


def _is_repeatable(early: list[Word], late: list[Word]) -> bool:
    if len(early) == 1:
        norm = _normalized(early[0])
        return (
            len(norm) > 1
            and not norm.isdigit()
            and norm not in NUMBER_WORDS
            and norm not in REDUPLICATION_EXCLUSIONS
        )
    return any(
        len(_normalized(w)) > 1 and not _normalized(w).isdigit() and _normalized(w) not in NUMBER_WORDS
        for w in (early + late)
    )


def _repeat_cuts(words: list[Word]) -> list[Cut]:
    cuts: list[Cut] = []
    claimed: set[int] = set()
    tokens = [_normalized(word) for word in words]
    for size in range(4, 0, -1):
        for index in range(len(words) - 2 * size + 1):
            early = words[index:index + size]
            late = words[index + size:index + 2 * size]
            if set(range(index, index + size)) & claimed:
                continue
            if tokens[index:index + size] == tokens[index + size:index + 2 * size] and _is_repeatable(early, late):
                cuts.append(Cut(
                    start=early[0].start,
                    end=early[-1].end,
                    reason="repeat",
                    text=" ".join(word.word.strip() for word in early),
                    confidence=CUT_CONFIDENCE["repeat"],
                ))
                claimed.update(range(index, index + size))
    return cuts


def _marker_cuts(words: list[Word], markers: list[dict]) -> list[Cut]:
    cuts: list[Cut] = []
    for marker in markers:
        if marker.get("type") != "bad_take":
            continue
        m_time = float(marker.get("time", 0.0))
        # Find start of phrase before marker (look back up to 25 seconds or previous pause)
        relevant_words = [w for w in words if w.end <= m_time and w.start >= m_time - 25.0]
        if not relevant_words:
            continue
        # Find if there was a pause > 0.5s before marker
        start_time = relevant_words[0].start
        for prev, curr in zip(relevant_words, relevant_words[1:]):
            if curr.start - prev.end > 0.5:
                start_time = curr.start
        cuts.append(
            Cut(
                start=round(start_time, 3),
                end=round(m_time, 3),
                reason="repeat",
                text="[bad_take marker]",
                confidence=1.0,
            )
        )
    return cuts


def _clamp_cuts_against_vad(cuts: list[Cut], speech: SpeechIntervals) -> list[Cut]:
    """Ensure silence cuts NEVER slice into speech intervals detected by Silero VAD."""
    if not speech.intervals:
        return cuts

    validated: list[Cut] = []
    for cut in cuts:
        if cut.reason != "silence":
            validated.append(cut)
            continue

        c_start, c_end = cut.start, cut.end
        overlap_speech = False
        for interval in speech.intervals:
            # Overlap condition
            if not (interval.end <= c_start or interval.start >= c_end):
                if interval.end > c_start and interval.start <= c_start:
                    c_start = max(c_start, interval.end + 0.02)
                if interval.start < c_end and interval.end >= c_end:
                    c_end = min(c_end, interval.start - 0.02)
                if interval.start > c_start and interval.end < c_end:
                    overlap_speech = True
                    break

        if not overlap_speech and c_end - c_start >= 0.05:
            validated.append(
                Cut(
                    start=round(c_start, 6),
                    end=round(c_end, 6),
                    reason=cut.reason,
                    text=cut.text,
                    confidence=cut.confidence,
                )
            )
    return validated


def analyze(
    transcript: Transcript,
    speech: SpeechIntervals,
    markers: list[dict] | None = None,
    screen: ScreenActivity | None = None,
    return_decisions: bool = False,
) -> list[Cut] | tuple[list[Cut], list[PauseDecision]]:
    """Return ordered cuts constrained by VAD speech boundaries, input activity, and screen motion."""
    words = _flatten(transcript)
    pause_cuts, pause_decisions = _evaluate_pauses(words, input_events=markers, screen=screen)
    raw_cuts = pause_cuts + _filler_cuts(words) + _repeat_cuts(words)
    if markers:
        raw_cuts.extend(_marker_cuts(words, markers))

    # Protect against cutting acoustic speech detected by VAD
    validated_cuts = _clamp_cuts_against_vad(raw_cuts, speech)
    sorted_cuts = sorted(validated_cuts, key=lambda cut: (cut.start, cut.end, cut.reason))
    if return_decisions:
        return sorted_cuts, pause_decisions
    return sorted_cuts

