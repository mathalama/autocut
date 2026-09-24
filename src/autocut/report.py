from pathlib import Path
from autocut.models import Cut, PauseDecision, Word


def format_context(cut: Cut, words: list[Word]) -> str:
    """Return 3 words before and 3 words after the cut, with cut text marked."""
    before_words = [w.word.strip() for w in words if w.end <= cut.start]
    before = before_words[-3:]

    after_words = [w.word.strip() for w in words if w.start >= cut.end]
    after = after_words[:3]

    center = f"[{cut.text}]" if cut.text else "[...]"
    return " ".join(before + [center] + after)


def write_cut_report(
    cuts: list[Cut],
    words: list[Word],
    output_path: Path,
    pause_decisions: list[PauseDecision] | None = None,
) -> None:
    """Write cuts and pause rationale sorted with context."""
    lines = ["# Cut report", ""]

    if pause_decisions:
        lines.extend([
            "## Pause Decisions (Demonstration vs Idle)",
            "",
            "| Timecode | Duration | Decision | Rationale / Activity Details |",
            "|---|---|---|---|",
        ])
        for p in pause_decisions:
            lines.append(f"| {p.start:.3f}–{p.end:.3f} | {p.duration:.2f}s | `{p.decision}` | {p.details} |")
        lines.append("")

    lines.extend([
        "## Scheduled Cuts",
        "",
        "| Timecode | Reason | Text | Context | Confidence |",
        "|---|---|---|---|---|",
    ])
    for cut in sorted(cuts, key=lambda item: (item.confidence, item.start, item.end)):
        timecode = f"{cut.start:.3f}–{cut.end:.3f}"
        reason = cut.reason
        text = cut.text or ""
        context = format_context(cut, words)
        confidence = f"{cut.confidence:.2f}"
        lines.append(f"| {timecode} | {reason} | {text} | {context} | {confidence} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

