"""Project words and transcript onto the edited timeline defined by EDL keep ranges."""

from autocut.models import KeepRange, Transcript, TranscriptSegment, Word


def remap_words(words: list[Word], keep: list[KeepRange]) -> list[Word]:
    """Project words onto the edited timeline, dropping words from cut sections."""
    if not keep:
        return []

    # Precalculate cumulative elapsed time before each keep segment
    offsets: list[float] = []
    total = 0.0
    for k in keep:
        offsets.append(total)
        total += (k.end - k.start)

    remapped: list[Word] = []
    for word in words:
        for k, offset in zip(keep, offsets):
            if word.end <= k.start or word.start >= k.end:
                continue

            # Clamping word boundaries to keep range
            start_in_keep = max(0.0, word.start - k.start)
            end_in_keep = min(k.end - k.start, word.end - k.start)

            new_start = offset + start_in_keep
            new_end = offset + end_in_keep

            if new_end > new_start + 0.02:  # Ignore micro fragments (< 20ms)
                remapped.append(
                    Word(
                        word=word.word,
                        start=round(new_start, 3),
                        end=round(new_end, 3),
                        probability=word.probability,
                    )
                )
            break
    return remapped


def group_words_into_cues(
    words: list[Word],
    max_words: int = 7,
    max_gap: float = 0.6,
    max_duration: float = 3.5,
) -> list[tuple[float, float, str, list[Word]]]:
    """Group sequential words into short subtitle lines for comfortable reading."""
    if not words:
        return []

    cues: list[tuple[float, float, str, list[Word]]] = []
    current_words: list[Word] = []

    for word in words:
        if not current_words:
            current_words.append(word)
            continue

        prev_word = current_words[-1]
        duration_if_added = word.end - current_words[0].start
        gap = word.start - prev_word.end

        should_split = (
            len(current_words) >= max_words
            or gap > max_gap
            or duration_if_added > max_duration
            or prev_word.word.endswith((".", "!", "?"))
        )

        if should_split:
            cue_text = " ".join(w.word.strip() for w in current_words)
            cues.append((current_words[0].start, current_words[-1].end, cue_text, list(current_words)))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        cue_text = " ".join(w.word.strip() for w in current_words)
        cues.append((current_words[0].start, current_words[-1].end, cue_text, list(current_words)))

    return cues
