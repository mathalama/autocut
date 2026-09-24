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


def generate_second_timeline(
    duration: float,
    cuts: list[Cut],
    speech: object = None,
    input_events: list[dict] | None = None,
    screen: object = None,
    output_path: Path | None = None,
) -> str:
    """Generate per-second timeline in the format: [MM:SS] VAD: ... | INPUT: ... | SCREEN: ... | DEC: ..."""
    lines = [
        "# Per-Second Timeline Debug",
        "# Format: [MM:SS] VAD | INPUT | SCREEN | DECISION",
        "-" * 95,
    ]
    total_seconds = int(duration) + 1

    events = input_events or []
    screen_intervals = getattr(screen, "intervals", []) if screen else []
    speech_intervals = getattr(speech, "intervals", []) if speech else []

    for sec in range(total_seconds):
        t_start = float(sec)
        t_end = float(sec + 1)
        mins, secs = divmod(sec, 60)
        timecode = f"{mins:02d}:{secs:02d}"

        # 1. VAD
        is_speech = any(not (sp.end <= t_start or sp.start >= t_end) for sp in speech_intervals)
        vad_str = "SPEECH " if is_speech else "SILENCE"

        # 2. Input
        sec_keys = sum(1 for e in events if e.get("type") == "key" and t_start <= float(e.get("time", -1)) < t_end)
        sec_clicks = sum(1 for e in events if e.get("type") == "click" and t_start <= float(e.get("time", -1)) < t_end)
        sec_markers = sum(1 for e in events if e.get("type") == "bad_take" and t_start <= float(e.get("time", -1)) < t_end)

        if sec_markers > 0:
            input_str = "MARKER(bad_take) "
        elif sec_keys > 0 and sec_clicks > 0:
            input_str = f"KEY({sec_keys})+CLK({sec_clicks}) "
        elif sec_keys > 0:
            input_str = f"KEY({sec_keys})           "
        elif sec_clicks > 0:
            input_str = f"CLICK({sec_clicks})         "
        else:
            input_str = "IDLE              "

        # 3. Screen
        active_screen = [sc for sc in screen_intervals if not (sc.end <= t_start or sc.start >= t_end)]
        if active_screen:
            screen_str = f"ACTIVE({active_screen[0].activity_type[:10]})"
        else:
            screen_str = "STATIC            "

        # 4. Decision: is this second cut?
        matching_cuts = [c for c in cuts if not (c.end <= t_start or c.start >= t_end)]
        if matching_cuts:
            cut = matching_cuts[0]
            txt = f": '{cut.text}'" if cut.text else ""
            dec_str = f"CUT ({cut.reason}{txt})"
        else:
            if is_speech:
                dec_str = "KEEP (speech)"
            elif sec_keys > 0 or sec_clicks > 0:
                dec_str = "KEEP (input)"
            elif active_screen:
                dec_str = "KEEP (screen)"
            else:
                dec_str = "KEEP (margin)"

        lines.append(f"[{timecode}] VAD: {vad_str} | INPUT: {input_str} | SCREEN: {screen_str} | DEC: {dec_str}")

    content = "\n".join(lines) + "\n"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    return content

