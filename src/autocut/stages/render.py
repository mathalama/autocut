"""Render EDL keep ranges using an ffmpeg filter-complex script."""

from pathlib import Path
import subprocess

from autocut.config import RENDER_ACROSSFADE_SECONDS, ensure_ffmpeg_in_path
from autocut.models import KeepRange

ensure_ffmpeg_in_path()


def filter_script(
    keep: list[KeepRange],
    audio_input: int,
    subtitles_path: Path | None = None,
) -> str:
    """Build trim/concat filters; audio fades do not overlap and keep A/V duration equal."""
    if not keep:
        raise ValueError("EDL contains no keep ranges.")
    lines = []
    for index, item in enumerate(keep):
        lines.append(f"[0:v]trim=start={item.start:.6f}:end={item.end:.6f},setpts=PTS-STARTPTS[v{index}]")
        lines.append(f"[{audio_input}:a]atrim=start={item.start:.6f}:end={item.end:.6f},asetpts=PTS-STARTPTS[a{index}]")

    video_final_out = "vout"
    if subtitles_path:
        sub_escaped = subtitles_path.as_posix().replace(":", r"\:")
        video_concat_out = "vpre" if len(keep) > 1 else "v0"
        sub_line = f"[{video_concat_out}]subtitles='{sub_escaped}'[vout]"
    else:
        sub_line = None

    if len(keep) == 1:
        if sub_line:
            lines.extend([sub_line, "[a0]anull[aout]"])
        else:
            lines.extend(["[v0]null[vout]", "[a0]anull[aout]"])
        return ";\n".join(lines)

    v_out_target = "vpre" if subtitles_path else "vout"
    lines.append("".join(f"[v{index}]" for index in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[{v_out_target}]")
    if sub_line:
        lines.append(sub_line)

    previous = "a0"
    for index in range(1, len(keep)):
        output = "aout" if index == len(keep) - 1 else f"ax{index}"
        lines.append(f"[{previous}][a{index}]acrossfade=d={RENDER_ACROSSFADE_SECONDS}:o=0[{output}]")
        previous = output
    return ";\n".join(lines)


def invert_keep_ranges(keep: list[KeepRange], duration: float) -> list[KeepRange]:
    """Invert keep ranges into removed ranges (the cuts)."""
    removed: list[KeepRange] = []
    cursor = 0.0
    for item in keep:
        if item.start > cursor:
            removed.append(KeepRange(start=round(cursor, 6), end=round(item.start, 6)))
        cursor = max(cursor, item.end)
    if cursor < duration:
        removed.append(KeepRange(start=round(cursor, 6), end=round(duration, 6)))
    return removed


def render_ranges(
    video: Path,
    audio: Path | None,
    keep: list[KeepRange],
    output: Path,
    script_path: Path,
    dry_run: bool = False,
    subtitles_path: Path | None = None,
) -> list[str]:
    """Write the script and, unless dry-run, invoke NVENC rendering."""
    if not keep:
        raise ValueError("EDL contains no keep ranges.")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    audio_input = 1 if audio else 0
    script_path.write_text(filter_script(keep, audio_input, subtitles_path=subtitles_path), encoding="utf-8")

    command = ["ffmpeg", "-y", "-i", str(video)]
    if audio:
        command.extend(["-i", str(audio)])
    command.extend([
        "-filter_complex_script", str(script_path), "-map", "[vout]", "-map", "[aout]",
        "-c:v", "h264_nvenc", "-c:a", "aac", str(output),
    ])
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        def _execute(cmd: list[str]) -> None:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as err:
                # If -filter_complex_script is unrecognized (e.g. ffmpeg 9.0+), retry with -filter_complex
                if "-filter_complex_script" in cmd and "filter_complex_script" in (err.stderr or ""):
                    new_cmd = list(cmd)
                    idx = new_cmd.index("-filter_complex_script")
                    new_cmd[idx] = "-filter_complex"
                    new_cmd[idx + 1] = script_path.read_text(encoding="utf-8")
                    subprocess.run(new_cmd, check=True)
                else:
                    raise

        try:
            _execute(command)
        except subprocess.CalledProcessError:
            fallback = list(command)
            if "h264_nvenc" in fallback:
                v_idx = fallback.index("h264_nvenc")
                fallback[v_idx] = "libx264"
                _execute(fallback)
            else:
                raise
    return command


