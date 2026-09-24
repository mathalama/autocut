"""Generate SRT and ASS subtitles from remapped word timestamps."""

from pathlib import Path
import pysubs2

from autocut.models import Word
from autocut.stages.remap import group_words_into_cues


def generate_subtitles(
    words: list[Word],
    srt_output_path: Path,
    ass_output_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Build SRT and styled ASS subtitle files from remapped words."""
    cues = group_words_into_cues(words)

    # 1. Standard SRT file
    srt_file = pysubs2.SSAFile()
    for start, end, text, _ in cues:
        start_ms = max(0, int(round(start * 1000)))
        end_ms = max(start_ms + 100, int(round(end * 1000)))
        srt_file.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))

    srt_output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_file.save(str(srt_output_path))

    # 2. Styled ASS file with word-level karaoke highlighting
    if ass_output_path is not None:
        ass_file = pysubs2.SSAFile()
        ass_file.info["PlayResX"] = "1920"
        ass_file.info["PlayResY"] = "1080"
        # Setup modern, legible subtitle style
        style = pysubs2.SSAStyle()
        style.fontname = "Segoe UI"
        style.fontsize = 54.0
        style.primarycolor = pysubs2.Color(255, 255, 255)       # White
        style.secondarycolor = pysubs2.Color(255, 215, 0)     # Golden highlight for karaoke
        style.outlinecolor = pysubs2.Color(16, 16, 16, 220)    # Dark outline
        style.backcolor = pysubs2.Color(0, 0, 0, 140)          # Semi-transparent shadow
        style.bold = True
        style.outline = 3.2
        style.shadow = 1.2
        style.alignment = pysubs2.Alignment.BOTTOM_CENTER
        style.marginv = 60

        ass_file.styles["Default"] = style

        for start, end, _, cue_words in cues:
            start_ms = max(0, int(round(start * 1000)))
            end_ms = max(start_ms + 100, int(round(end * 1000)))

            # Build karaoke tags: \k<duration_in_centiseconds>word
            karaoke_parts = []
            for w in cue_words:
                duration_cs = max(1, int(round((w.end - w.start) * 100)))
                karaoke_parts.append(f"{{\\k{duration_cs}}}{w.word.strip()}")

            styled_text = " ".join(karaoke_parts)
            ass_file.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=styled_text))

        ass_output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_file.save(str(ass_output_path))

    return srt_output_path, ass_output_path
