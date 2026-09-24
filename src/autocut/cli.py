"""Typer commands that compose the autocut screencast editing pipeline."""

import json
from pathlib import Path
import shutil
from typing import Optional

import typer

from autocut.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_INITIAL_PROMPT, DEFAULT_MODEL
from autocut.models import Cut, EditDecisionList, MediaInfo, ScreenActivity, SpeechIntervals, Transcript
from autocut.report import generate_second_timeline, write_cut_report
from autocut.stages.analyze import analyze as analyze_stage
from autocut.stages.asr import transcribe as transcribe_audio
from autocut.stages.edl import build_edl
from autocut.stages.ingest import extract_audio, normalize_to_cfr, probe_media
from autocut.stages.remap import remap_words
from autocut.stages.render import invert_keep_ranges, render_ranges
from autocut.stages.screen import detect_screen_activity
from autocut.stages.subtitles import generate_subtitles
from autocut.stages.vad import detect_speech

app = typer.Typer(help="CLI tool for non-destructive screencast editing.")


def _default_work_dir(video: Path) -> Path:
    return Path("work") / video.stem


def _load_markers(markers_path: Optional[Path], work_dir: Path) -> Optional[list[dict]]:
    path = markers_path if (markers_path and markers_path.exists()) else (work_dir / "markers.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _ingest(video: Path, work_dir: Path, audio: Optional[Path]) -> tuple[Path, Path]:
    """Create normalized audio and media metadata in one work directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    media = probe_media(video)
    (work_dir / "ingest.json").write_text(media.model_dump_json(indent=2), encoding="utf-8")
    video_source = video
    if media.is_vfr:
        typer.echo("VFR detected in source video; normalizing to CFR...")
        video_source = normalize_to_cfr(video, work_dir / "normalized_cfr.mp4", fps=media.fps or 60.0)
    audio_path = extract_audio(audio or video_source, work_dir / "audio.wav")
    return video_source, audio_path


@app.command()
def ingest(
    video: Path = typer.Argument(..., exists=True, readable=True),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    audio: Optional[Path] = typer.Option(None, "--audio", exists=True, readable=True),
) -> None:
    """Probe VIDEO and extract its (or --audio's) normalized WAV."""
    _, audio_path = _ingest(video, work_dir or _default_work_dir(video), audio)
    typer.echo(audio_path)


@app.command()
def vad(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Produce speech.json from a work directory's audio.wav."""
    result = detect_speech(work_dir / "audio.wav", work_dir / "speech.json")
    typer.echo(f"{len(result.intervals)} speech intervals")


@app.command(name="transcribe")
def transcribe_command(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device"),
    compute_type: str = typer.Option(DEFAULT_COMPUTE_TYPE, "--compute-type"),
    language: Optional[str] = typer.Option(None, "--language"),
    initial_prompt: Optional[str] = typer.Option(DEFAULT_INITIAL_PROMPT, "--initial-prompt"),
) -> None:
    """Create timestamped transcript.json from speech.json."""
    intervals = SpeechIntervals.model_validate_json(
        (work_dir / "speech.json").read_text(encoding="utf-8")
    )
    result = transcribe_audio(
        audio_path=work_dir / "audio.wav",
        intervals=intervals.intervals,
        output_path=work_dir / "transcript.json",
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
        initial_prompt=initial_prompt,
    )
    typer.echo(f"{len(result.segments)} transcript segments")


@app.command()
def screen(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    fps: int = typer.Option(5, "--fps", help="Sampling fps for screen frame diffs."),
) -> None:
    """Detect screen activity (typing, terminal scroll) and output screen.json."""
    media = MediaInfo.model_validate_json((work_dir / "ingest.json").read_text(encoding="utf-8"))
    video_source = (work_dir / "normalized_cfr.mp4") if (work_dir / "normalized_cfr.mp4").exists() else media.source
    activity = detect_screen_activity(video_source, work_dir / "screen.json", fps=fps)
    typer.echo(f"Found {len(activity.intervals)} screen activity intervals")


@app.command()
def analyze(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    markers: Optional[Path] = typer.Option(None, "--markers", "-m", help="Path to markers.json with bad_take cuts."),
) -> None:
    """Analyze pauses, fillers, and repeats to produce cuts.json and cuts_report.md."""
    transcript = Transcript.model_validate_json(
        (work_dir / "transcript.json").read_text(encoding="utf-8")
    )
    speech = SpeechIntervals.model_validate_json(
        (work_dir / "speech.json").read_text(encoding="utf-8")
    )
    markers_data = _load_markers(markers, work_dir)
    screen_path = work_dir / "screen.json"
    screen_data = (
        ScreenActivity.model_validate_json(screen_path.read_text(encoding="utf-8"))
        if screen_path.exists()
        else None
    )
    cuts, pause_decisions = analyze_stage(transcript, speech, markers=markers_data, screen=screen_data, return_decisions=True)
    cuts_json = [cut.model_dump() for cut in cuts]
    (work_dir / "cuts.json").write_text(json.dumps(cuts_json, indent=2), encoding="utf-8")

    all_words = [word for segment in transcript.segments for word in segment.words]
    write_cut_report(cuts, all_words, work_dir / "cuts_report.md", pause_decisions=pause_decisions)

    ingest_path = work_dir / "ingest.json"
    duration = (
        MediaInfo.model_validate_json(ingest_path.read_text(encoding="utf-8")).duration
        if ingest_path.exists()
        else (all_words[-1].end if all_words else 0.0)
    )
    generate_second_timeline(
        duration=duration,
        cuts=cuts,
        speech=speech,
        input_events=markers_data,
        screen=screen_data,
        output_path=work_dir / "timeline_debug.txt",
    )

    silence_count = sum(1 for c in cuts if c.reason == "silence")
    filler_count = sum(1 for c in cuts if c.reason == "filler")
    repeat_count = sum(1 for c in cuts if c.reason == "repeat")
    typer.echo(
        f"Found {len(cuts)} cuts: {silence_count} silence, {filler_count} filler, {repeat_count} repeat"
    )


@app.command()
def subtitles(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    ass: bool = typer.Option(True, "--ass/--no-ass", help="Generate styled ASS subtitles in addition to SRT."),
) -> None:
    """Generate remapped SRT (and styled ASS) subtitles based on edl.json."""
    transcript = Transcript.model_validate_json((work_dir / "transcript.json").read_text(encoding="utf-8"))
    edl_path = work_dir / "edl.json"
    if not edl_path.exists():
        typer.echo("Error: edl.json not found. Run 'autocut render --dry-run' or 'autocut render' first.", err=True)
        raise typer.Exit(code=1)
    edl = EditDecisionList.model_validate_json(edl_path.read_text(encoding="utf-8"))
    all_words = [w for s in transcript.segments for w in s.words]
    remapped = remap_words(all_words, edl.keep)
    srt_path = work_dir / "subtitles.srt"
    ass_path = (work_dir / "subtitles.ass") if ass else None
    generate_subtitles(remapped, srt_path, ass_path)
    typer.echo(f"Generated subtitles: {srt_path}")
    if ass_path:
        typer.echo(f"Generated styled ASS subtitles: {ass_path}")


@app.command()
def render(
    work_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write the edited video."),
    markers: Optional[Path] = typer.Option(None, "--markers", "-m", help="Path to markers.json with bad_take cuts."),
    export_removed: bool = typer.Option(False, "--export-removed", help="Export a video of only removed intervals."),
    with_subtitles: bool = typer.Option(True, "--subtitles/--no-subtitles", help="Generate remapped subtitles (SRT & ASS)."),
    burn_subtitles: bool = typer.Option(False, "--burn-subtitles", help="Burn remapped subtitles directly into edited video."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Calculate duration and cuts summary without rendering."),
) -> None:
    """Build EDL, generate remapped subtitles, and render the edited video (or dry-run)."""
    media = MediaInfo.model_validate_json(
        (work_dir / "ingest.json").read_text(encoding="utf-8")
    )
    transcript = Transcript.model_validate_json(
        (work_dir / "transcript.json").read_text(encoding="utf-8")
    )
    cuts_path = work_dir / "cuts.json"
    if not cuts_path.exists():
        speech = SpeechIntervals.model_validate_json(
            (work_dir / "speech.json").read_text(encoding="utf-8")
        )
        markers_data = _load_markers(markers, work_dir)
        screen_path = work_dir / "screen.json"
        screen_data = (
            ScreenActivity.model_validate_json(screen_path.read_text(encoding="utf-8"))
            if screen_path.exists()
            else None
        )
        cuts, pause_decisions = analyze_stage(transcript, speech, markers=markers_data, screen=screen_data, return_decisions=True)
        cuts_path.write_text(json.dumps([c.model_dump() for c in cuts], indent=2), encoding="utf-8")
        all_words = [w for seg in transcript.segments for w in seg.words]
        write_cut_report(cuts, all_words, work_dir / "cuts_report.md", pause_decisions=pause_decisions)
        generate_second_timeline(
            duration=media.duration,
            cuts=cuts,
            speech=speech,
            input_events=markers_data,
            screen=screen_data,
            output_path=work_dir / "timeline_debug.txt",
        )
    else:
        cuts = [Cut.model_validate(c) for c in json.loads(cuts_path.read_text(encoding="utf-8"))]

    all_words = [word for segment in transcript.segments for word in segment.words]
    video_source = work_dir / "normalized_cfr.mp4" if (work_dir / "normalized_cfr.mp4").exists() else media.source
    edl = build_edl(
        video=video_source,
        audio=work_dir / "audio.wav",
        duration=media.duration,
        cuts=cuts,
        words=all_words,
    )
    (work_dir / "edl.json").write_text(edl.model_dump_json(indent=2), encoding="utf-8")

    orig_duration = media.duration
    kept_duration = sum(item.end - item.start for item in edl.keep)
    removed_duration = orig_duration - kept_duration

    silence_count = sum(1 for c in cuts if c.reason == "silence")
    filler_count = sum(1 for c in cuts if c.reason == "filler")
    repeat_count = sum(1 for c in cuts if c.reason == "repeat")

    typer.echo(f"Original duration: {orig_duration:.2f}s")
    typer.echo(f"Edited duration:   {kept_duration:.2f}s")
    typer.echo(f"Removed duration:  {removed_duration:.2f}s")
    typer.echo(f"Total cuts: {len(cuts)}")
    typer.echo(f"  - silence: {silence_count}")
    typer.echo(f"  - filler:  {filler_count}")
    typer.echo(f"  - repeat:  {repeat_count}")

    if with_subtitles or burn_subtitles:
        remapped_words = remap_words(all_words, edl.keep)
        srt_path = work_dir / "subtitles.srt"
        ass_path = work_dir / "subtitles.ass"
        generate_subtitles(remapped_words, srt_path, ass_path)
        typer.echo(f"Generated remapped subtitles: {srt_path.name}, {ass_path.name}")

    if dry_run:
        typer.echo("Dry-run complete: no video was rendered.")
        return

    out_file = output or (work_dir / "edited.mp4")
    script_path = work_dir / "filter_script.txt"
    sub_burn = (work_dir / "subtitles.ass") if burn_subtitles else None
    render_ranges(
        video=video_source,
        audio=None,
        keep=edl.keep,
        output=out_file,
        script_path=script_path,
        dry_run=False,
        subtitles_path=sub_burn,
    )
    typer.echo(f"Rendered edited video: {out_file}")

    if export_removed:
        removed_keep = invert_keep_ranges(edl.keep, media.duration)
        if removed_keep:
            rem_output = (
                output.parent / f"{output.stem}_removed{output.suffix}"
                if output
                else (work_dir / "removed.mp4")
            )
            rem_script = work_dir / "filter_script_removed.txt"
            render_ranges(
                video=video_source,
                audio=None,
                keep=removed_keep,
                output=rem_output,
                script_path=rem_script,
                dry_run=False,
            )
            typer.echo(f"Rendered removed intervals video: {rem_output}")
        else:
            typer.echo("No cuts to export in removed video.")


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True, readable=True),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    audio: Optional[Path] = typer.Option(None, "--audio", exists=True, readable=True),
    markers: Optional[Path] = typer.Option(None, "--markers", "-m", exists=True, readable=True),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device"),
    compute_type: str = typer.Option(DEFAULT_COMPUTE_TYPE, "--compute-type"),
    language: Optional[str] = typer.Option(None, "--language"),
    initial_prompt: Optional[str] = typer.Option(DEFAULT_INITIAL_PROMPT, "--initial-prompt"),
    do_render: bool = typer.Option(False, "--render", help="Also run analyze and render to produce edited.mp4."),
    export_removed: bool = typer.Option(False, "--export-removed", help="Export removed intervals when rendering."),
    with_subtitles: bool = typer.Option(True, "--subtitles/--no-subtitles", help="Generate remapped subtitles."),
    burn_subtitles: bool = typer.Option(False, "--burn-subtitles", help="Burn subtitles into rendered video."),
    detect_screen: bool = typer.Option(True, "--detect-screen/--no-detect-screen", help="Detect screen activity to protect silent demonstrations."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Calculate duration and cuts summary without rendering."),
) -> None:
    """Run full autocut pipeline: ingest, screen activity, VAD, ASR, and optionally analyze/render."""
    output_dir = work_dir or _default_work_dir(video)
    output_dir.mkdir(parents=True, exist_ok=True)
    if markers and markers.exists():
        shutil.copy2(markers, output_dir / "markers.json")
    video_source, audio_path = _ingest(video, output_dir, audio)
    if detect_screen:
        detect_screen_activity(video_source, output_dir / "screen.json")
    intervals = detect_speech(audio_path, output_dir / "speech.json")
    result = transcribe_audio(
        audio_path=audio_path,
        intervals=intervals.intervals,
        output_path=output_dir / "transcript.json",
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
        initial_prompt=initial_prompt,
    )
    if do_render or dry_run:
        render(
            work_dir=output_dir,
            output=None,
            markers=markers,
            export_removed=export_removed,
            with_subtitles=with_subtitles,
            burn_subtitles=burn_subtitles,
            dry_run=dry_run,
        )
    else:
        typer.echo(json.dumps({"work_dir": str(output_dir), "segments": len(result.segments)}))


@app.command(name="listen-markers")
def listen_markers_command(
    output: Path = typer.Option(Path("work/markers.json"), "--output", "-o", help="Path to markers.json file."),
    key: str = typer.Option("f12", "--key", "-k", help="Hotkey for bad take marker (f12, scroll_lock, pause, etc.)."),
    sync_key: str = typer.Option("scroll_lock", "--sync-key", "-s", help="Hotkey for manual start sync (scroll_lock, f11, etc.)."),
    port: int = typer.Option(4455, "--port", "-p", help="OBS WebSocket port."),
    password: str = typer.Option("", "--password", help="OBS WebSocket password."),
) -> None:
    """Listen for global hotkey and anonymous user input during recording."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.listen_markers import listen
    listen(output, bad_take_key=key, sync_key=sync_key, obs_port=port, obs_password=password)


def main() -> None:
    """Invoke the CLI entry point installed by the package metadata."""
    app()


if __name__ == "__main__":
    main()


