"""Typer CLI commands for real-time autocut subtitles."""

from pathlib import Path
import time
from typing import Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from autocut.audio import MicrophoneAccessError, MicrophoneRecorder, list_input_devices
from autocut.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_PORT,
)
from autocut.engine import RealtimeSubtitleEngine
from autocut.recorder import CaptionRecorder
from autocut.server import OverlayServer
from autocut.transcriber import BatchTranscriber, save_float32_as_wav
from autocut.stt import ModelPool, STTSegment
from autocut.stt.rolling import RollingTranscriber
from autocut.daemon import STTDaemon, DEFAULT_DAEMON_PORT, DaemonClient

app = typer.Typer(
    name="autocut",
    help="Real-time AI subtitles and captions engine for OBS and live recordings.",
    add_completion=False,
)
console = Console()


@app.command(name="devices")
def list_devices_command() -> None:
    """List all available microphone input devices."""
    devices = list_input_devices()
    if not devices:
        console.print("[yellow]No audio input devices found.[/yellow]")
        return

    table = Table(title="Available Audio Input Devices", header_style="bold cyan")
    table.add_column("Index", justify="right", style="cyan", no_wrap=True)
    table.add_column("Device Name", style="white")
    table.add_column("Channels", justify="center", style="green")
    table.add_column("Sample Rate", justify="right", style="magenta")
    table.add_column("Default", justify="center", style="yellow")

    for d in devices:
        table.add_row(
            str(d["index"]),
            d["name"],
            str(d["channels"]),
            f"{int(d['default_samplerate'])} Hz",
            "[bold green]YES[/]" if d["is_default"] else "",
        )

    console.print(table)


@app.command(name="live")
def live_command(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Whisper model name (small.en, tiny.en, base.en, etc.)."),
    device: str = typer.Option(DEFAULT_DEVICE, "--device", "-d", help="Inference device: 'cuda' or 'cpu'."),
    compute_type: str = typer.Option(DEFAULT_COMPUTE_TYPE, "--compute-type", help="Quantization: 'float16', 'int8', etc."),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", "-l", help="Language code (e.g. 'ru', 'en', or 'auto')."),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Unified HTTP and WebSocket server port for OBS Browser Source."),
    ws_port: Optional[int] = typer.Option(None, "--ws-port", help="Deprecated: Server now runs unified on single port."),
    device_index: Optional[int] = typer.Option(None, "--device-index", help="Microphone index from 'autocut devices'."),
    output_srt: Optional[Path] = typer.Option(Path("captions.srt"), "--output-srt", "-o", help="File to record subtitles in SRT format."),
    energy_threshold: float = typer.Option(0.015, "--energy-threshold", help="Microphone noise gate threshold (increase to 0.02-0.03 for noisy mics)."),
    translate: bool = typer.Option(False, "--translate", "-T", help="Translate speech to English live on overlay."),
    hotkey: Optional[str] = typer.Option("f9", "--hotkey", "-k", help="Global hotkey to toggle pause/mute (e.g. 'f9', 'pause')."),
    theme: str = typer.Option("standard", "--theme", "-t", help="Overlay theme: 'standard' (white on black), 'black' (black on white), 'outline' (no bg)."),
    size: int = typer.Option(28, "--size", "-s", help="Font size in pixels for the overlay."),
    engine: str = typer.Option("whisper", "--engine", "-e", help="STT engine to use: 'whisper' (faster-whisper) or 'higgs' (bosonai/higgs-audio-v3-stt)."),
) -> None:
    """Start real-time subtitle generation with OBS Browser Source overlay."""
    actual_port = ws_port if ws_port else port
    obs_url = f"http://localhost:{actual_port}/?theme={theme}&size={size}"
    mode_text = "[bold magenta]Real-Time Translation (→ English)[/]" if translate else "[bold cyan]Live Transcription[/]"
    hotkey_text = f"[bold yellow]{hotkey.upper()}[/] (Toggle Mute)" if hotkey else "[dim]None[/dim]"

    # If translation is requested and model is .en, switch to multilingual model
    actual_model = model
    if translate and actual_model.endswith(".en"):
        actual_model = actual_model[:-3]

    engine_display = f"Higgs Audio v3 STT ({actual_model})" if engine.lower() == "higgs" else f"Whisper ({actual_model})"
    console.print(
        Panel.fit(
            f"[bold green]AutoCut Real-Time Subtitles[/bold green]\n\n"
            f"[bold]Engine:[/] [cyan]{engine_display}[/]   "
            f"[bold]Mode:[/] {mode_text}\n"
            f"[bold]Hotkey:[/] {hotkey_text}   "
            f"[bold]Language:[/] [cyan]{language}[/]   "
            f"[bold]Device:[/] [cyan]{device} ({compute_type})[/]\n"
            f"[bold]OBS Browser Source URL:[/] [bold underline yellow]{obs_url}[/]\n"
            f"[bold]Saving Subtitles to:[/] [cyan]{output_srt}[/]\n\n"
            f"[dim]Press {hotkey.upper() if hotkey else 'Ctrl+C'} to pause/mute. Press Ctrl+C to exit.[/dim]",
            title="Single-Pass Subtitle Engine Active",
            border_style="cyan",
        )
    )

    recorder = CaptionRecorder(srt_path=output_srt, txt_path=output_srt.with_suffix(".txt") if output_srt else None)
    server = OverlayServer(host="0.0.0.0", port=actual_port)
    server.start()

    session_start = time.time()
    last_caption_time = session_start

    def on_caption(text: str, is_final: bool = True) -> None:
        nonlocal last_caption_time
        now = time.time()
        start_rel = max(0.0, last_caption_time - session_start)
        end_rel = max(start_rel + 0.5, now - session_start)

        # Broadcast final subtitle directly to OBS Browser Source
        server.broadcast({
            "type": "caption",
            "text": text,
            "is_final": True,
        })

        console.print(f"[bold cyan][SUB][/] [white]{text}[/]")
        recorder.add_caption(text, start_rel, end_rel)
        last_caption_time = now

    def on_clear() -> None:
        server.broadcast({"type": "clear"})

    def on_status(payload: dict[str, Any]) -> None:
        server.broadcast({"type": "status", **payload})

    engine = RealtimeSubtitleEngine(
        model_name=actual_model,
        device=device,
        compute_type=compute_type,
        language=language,
        task="translate" if translate else "transcribe",
        engine_type=engine,
        device_index=device_index,
        energy_threshold=energy_threshold,
        on_caption=on_caption,
        on_clear=on_clear,
        on_status=on_status,
    )

    if hotkey:
        try:
            import keyboard
            def _toggle():
                paused = engine.toggle_pause()
                if paused:
                    console.print(f"\n[bold yellow][PAUSED] Subtitles[/] (Press {hotkey.upper()} to resume)")
                else:
                    console.print(f"\n[bold green][RESUMED] Subtitles[/]")
            keyboard.add_hotkey(hotkey, _toggle)
        except Exception as e:
            logger.debug("Could not register global hotkey %s: %s", hotkey, e)

    console.print("[dim]Loading model and starting audio listener...[/dim]")
    try:
        engine.start()
    except MicrophoneAccessError as e:
        console.print(f"[bold red][ERROR][/bold red] {e}")
        server.stop()
        raise typer.Exit(1)
    console.print("[bold green][OK] Listening to microphone. Speak now...[/bold green]\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live subtitles...[/yellow]")
    finally:
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        engine.stop()
        server.stop()
        console.print(f"[bold green][OK] Subtitles saved to [cyan]{output_srt}[/cyan].[/bold green]")


@app.command(name="cut")
def cut_command(
    input_video: Path = typer.Argument(..., help="Path to input video or audio file (MP4, MKV, MOV, WAV)."),
    output_video: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to export trimmed video file."),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model name for transcription (base, small, etc.)."),
    device: str = typer.Option(DEFAULT_DEVICE, "--device", "-d", help="Inference device: 'cuda' or 'cpu'."),
    compute_type: str = typer.Option(DEFAULT_COMPUTE_TYPE, "--compute-type", help="Quantization: 'float16', 'int8', etc."),
    pause_threshold: float = typer.Option(0.6, "--pause-threshold", "-p", help="Minimum silence duration in seconds to cut out."),
    margin: float = typer.Option(0.15, "--margin", help="Audio padding margin around speech in seconds."),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g. 'en', 'ru')."),
    output_srt: Optional[Path] = typer.Option(None, "--output-srt", help="Optional path to export synchronized subtitles."),
) -> None:
    """Automatically cut silent pauses and dead air from recorded videos."""
    if not input_video.exists():
        console.print(f"[bold red]Error:[/] Input file '{input_video}' does not exist.")
        raise typer.Exit(1)

    actual_output = output_video or input_video.with_stem(f"{input_video.stem}_cut")

    console.print(
        Panel.fit(
            f"[bold green]AutoCut Video Silence Cutter[/bold green]\n\n"
            f"[bold]Input Video:[/] [cyan]{input_video}[/]\n"
            f"[bold]Output Video:[/] [cyan]{actual_output}[/]\n"
            f"[bold]Silence Threshold:[/] [yellow]>= {pause_threshold}s[/]   "
            f"[bold]Speech Margin:[/] [yellow]±{margin}s[/]\n"
            f"[bold]Whisper Model:[/] [cyan]{model}[/] ({device})",
            title="AutoCut Processing",
            border_style="cyan",
        )
    )

    from autocut.cut import cut_video, format_duration

    with console.status("[bold cyan]Analyzing audio and detecting pauses...[/bold cyan]") as status:
        def update_progress(msg: str):
            status.update(f"[bold cyan]{msg}[/bold cyan]")

        try:
            stats = cut_video(
                input_file=input_video,
                output_file=actual_output,
                model_name=model,
                device=device,
                compute_type=compute_type,
                pause_threshold=pause_threshold,
                margin=margin,
                language=language,
                output_srt=output_srt,
                progress_callback=update_progress,
            )
        except Exception as e:
            console.print(f"[bold red]Failed to cut video:[/] {e}")
            raise typer.Exit(1)

    orig_str = format_duration(stats["original_duration"])
    cut_str = format_duration(stats["cut_duration"])
    saved_str = format_duration(stats["time_saved"])
    pct = stats["reduction_percent"]
    segs = stats["num_segments"]

    console.print(
        Panel.fit(
            f"[bold green]AutoCut Completed Successfully![/bold green]\n\n"
            f"[bold]Original Duration:[/] {orig_str}\n"
            f"[bold]Trimmed Duration:[/]  [bold cyan]{cut_str}[/]\n"
            f"[bold]Time Cut Away:[/]     [bold yellow]{saved_str}[/] ([bold green]-{pct:.1f}%[/])\n"
            f"[bold]Kept Segments:[/]     {segs}\n"
            f"[bold]Saved File:[/]        [underline green]{actual_output}[/]",
            title="Cut Summary",
            border_style="green",
        )
    )


def copy_to_clipboard(text: str) -> bool:
    """Copy text string to Windows clipboard."""
    import subprocess
    try:
        proc = subprocess.Popen(["powershell", "-Command", "$input | Set-Clipboard"], stdin=subprocess.PIPE)
        proc.communicate(input=text.encode("utf-8"), timeout=3)
        return proc.returncode == 0
    except Exception:
        return False


@app.command(name="record")
def record_command(
    output_srt: Path = typer.Option(Path("captions.srt"), "--output-srt", "-o", help="Path to export SRT subtitles."),
    engine: str = typer.Option("higgs", "--engine", "-e", help="STT engine: 'higgs' (bosonai/higgs-audio-v3-stt) or 'whisper'."),
    model: str = typer.Option("large-v3-turbo", "--model", "-m", help="Whisper model name (if engine=whisper)."),
    device: str = typer.Option("cuda", "--device", "-d", help="Inference device: 'cuda' or 'cpu'."),
    compute_type: str = typer.Option("float16", "--compute-type", help="Quantization type."),
    language: str = typer.Option("auto", "--language", "-l", help="Language code (e.g. 'ru', 'en', 'auto')."),
    hotkey: Optional[str] = typer.Option("f9", "--hotkey", "-k", help="Hotkey to stop recording."),
    device_index: Optional[int] = typer.Option(None, "--device-index", help="Microphone index from 'autocut devices'."),
    save_wav: Optional[Path] = typer.Option(None, "--save-wav", "-w", help="Optional path to save recorded audio WAV."),
    clipboard: bool = typer.Option(True, "--clipboard/--no-clipboard", help="Copy full transcribed text to clipboard."),
) -> None:
    """Record speech from microphone, then stop and generate high-accuracy subtitles with Higgs STT or Whisper."""
    engine_name = "Higgs Audio v3 STT (bosonai)" if engine.lower() == "higgs" else f"Whisper ({model})"
    stop_hint = f"Press {hotkey.upper()} or ENTER" if hotkey else "Press ENTER"

    console.print(
        Panel.fit(
            f"[bold green]AutoCut High-Accuracy Studio Recorder[/bold green]\n\n"
            f"[bold]STT Engine:[/]  [cyan]{engine_name}[/]\n"
            f"[bold]Language:[/]    [cyan]{language}[/]   [bold]Device:[/] [cyan]{device} ({compute_type})[/]\n"
            f"[bold]Output SRT:[/]  [cyan]{output_srt}[/]\n"
            f"[bold]Output TXT:[/]  [cyan]{output_srt.with_suffix('.txt')}[/]\n\n"
            f"[bold yellow]{stop_hint} when finished speaking to transcribe.[/bold yellow]",
            title="Studio Recording Mode",
            border_style="magenta",
        )
    )

    console.print("[dim]Pre-warming model for rolling real-time transcription...[/dim]")
    engine_instance = ModelPool.instance().get_engine(
        engine_type=engine,
        model_name=model if engine.lower() != "higgs" else None,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    rolling = RollingTranscriber(
        engine=engine_instance,
        sample_rate=16000,
        pause_threshold=0.45,
        energy_threshold=0.015,
    )
    rolling.start()

    recorder = MicrophoneRecorder(device_index=device_index, on_chunk=rolling.process_audio_chunk)
    try:
        recorder.start()
    except MicrophoneAccessError as e:
        console.print(f"[bold red][ERROR][/bold red] {e}")
        rolling.finish()
        raise typer.Exit(1)
    start_time = time.time()
    console.print(f"\n[bold red][RECORDING ACTIVE][/bold red] Speak now. ({stop_hint} to finish)\n")

    stop_event = threading.Event()

    def _trigger_stop():
        stop_event.set()

    if hotkey:
        try:
            import keyboard
            keyboard.add_hotkey(hotkey, _trigger_stop)
        except Exception:
            pass

    # Wait for either hotkey or user pressing ENTER in terminal
    try:
        import sys
        if sys.stdin and sys.stdin.isatty():
            import threading
            def _wait_stdin():
                try:
                    sys.stdin.readline()
                    stop_event.set()
                except Exception:
                    pass
            t = threading.Thread(target=_wait_stdin, daemon=True)
            t.start()

        while not stop_event.is_set():
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r  [REC] {mins:02d}:{secs:02d} ... ({stop_hint} to stop)")
            sys.stdout.flush()
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    console.print("[yellow]Recording stopped. Assembling pre-transcribed subtitles...[/yellow]")
    t0 = time.time()
    audio = recorder.stop()
    duration_sec = len(audio) / 16000.0
    console.print(f"[dim]Captured {duration_sec:.1f}s of audio.[/dim]")

    if len(audio) < 16000 * 0.5:
        console.print("[yellow]Audio too short (< 0.5s). No transcription generated.[/yellow]")
        return

    if save_wav:
        save_float32_as_wav(audio, save_wav)
        console.print(f"[dim]Saved raw recording to [cyan]{save_wav}[/cyan][/dim]")

    # Retrieve pre-emptively transcribed segments from rolling pipeline
    raw_segments = rolling.finish()
    inf_time = time.time() - t0

    # If rolling VAD found no distinct pauses or user spoke in one breath, run batch on full audio
    if not raw_segments:
        raw_segments = engine_instance.transcribe(audio, sample_rate=16000)
        inf_time = time.time() - t0

    if not raw_segments:
        console.print("[yellow]No speech detected in recording.[/yellow]")
        return

    segments = [s.to_dict() for s in raw_segments]
    txt_path = output_srt.with_suffix(".txt")
    rec = CaptionRecorder(srt_path=output_srt, txt_path=txt_path)
    for s in segments:
        rec.add_caption(s["text"], s["start"], s["end"])

    full_text = " ".join(s["text"] for s in segments)
    if clipboard:
        copied = copy_to_clipboard(full_text)
        clip_note = " [bold green](Copied to Clipboard!)[/]" if copied else ""
    else:
        clip_note = ""

    table = Table(title=f"Transcribed Subtitles ({len(segments)} segments in {inf_time:.1f}s)", header_style="bold cyan")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Time Range", style="magenta", no_wrap=True)
    table.add_column("Text", style="white")

    for s in segments:
        mins_s, secs_s = divmod(int(s["start"]), 60)
        mins_e, secs_e = divmod(int(s["end"]), 60)
        time_range = f"{mins_s:02d}:{secs_s:02d} - {mins_e:02d}:{secs_e:02d}"
        table.add_row(str(s["index"]), time_range, s["text"])

    console.print(table)
    console.print(f"\n[bold green]Saved subtitles to [cyan]{output_srt}[/cyan] and [cyan]{txt_path}[/cyan]{clip_note}[/bold green]\n")


@app.command(name="transcribe")
def transcribe_command(
    input_file: Path = typer.Argument(..., help="Path to audio or video file (MP4, MKV, WAV, MP3)."),
    output_srt: Optional[Path] = typer.Option(None, "--output-srt", "-o", help="Path to export SRT subtitles."),
    engine: str = typer.Option("higgs", "--engine", "-e", help="STT engine: 'higgs' (bosonai/higgs-audio-v3-stt) or 'whisper'."),
    model: str = typer.Option("large-v3-turbo", "--model", "-m", help="Whisper model name (if engine=whisper)."),
    device: str = typer.Option("cuda", "--device", "-d", help="Inference device: 'cuda' or 'cpu'."),
    compute_type: str = typer.Option("float16", "--compute-type", help="Quantization type."),
    language: str = typer.Option("auto", "--language", "-l", help="Language code (e.g. 'ru', 'en', 'auto')."),
    clipboard: bool = typer.Option(False, "--clipboard", help="Copy full text to Windows clipboard."),
) -> None:
    """Transcribe an existing video or audio file directly into high-accuracy SRT and TXT subtitles."""
    if not input_file.exists():
        console.print(f"[bold red]File not found:[/] {input_file}")
        raise typer.Exit(1)

    actual_srt = output_srt or input_file.with_suffix(".srt")
    actual_txt = actual_srt.with_suffix(".txt")
    engine_name = "Higgs Audio v3 STT" if engine.lower() == "higgs" else f"Whisper ({model})"

    console.print(
        Panel.fit(
            f"[bold green]AutoCut Video/Audio Subtitle Generator[/bold green]\n\n"
            f"[bold]Input File:[/]  [cyan]{input_file}[/]\n"
            f"[bold]STT Engine:[/]  [cyan]{engine_name}[/]\n"
            f"[bold]Language:[/]    [cyan]{language}[/]   [bold]Device:[/] [cyan]{device} ({compute_type})[/]\n"
            f"[bold]Output SRT:[/]  [underline yellow]{actual_srt}[/]",
            title="Batch File Transcription",
            border_style="cyan",
        )
    )

    console.print("[dim]Extracting audio and transcribing...[/dim]")
    transcriber = BatchTranscriber(
        engine_type=engine,
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    t0 = time.time()
    segments = transcriber.transcribe_file(input_file, srt_path=actual_srt, txt_path=actual_txt)
    inf_time = time.time() - t0

    if not segments:
        console.print("[yellow]No speech detected in media file.[/yellow]")
        return

    full_text = " ".join(s["text"] for s in segments)
    if clipboard:
        copy_to_clipboard(full_text)

    console.print(f"[bold green]Successfully generated {len(segments)} subtitle segments in {inf_time:.1f}s![/bold green]")
    console.print(f"  SRT: [cyan]{actual_srt}[/cyan]")
    console.print(f"  TXT: [cyan]{actual_txt}[/cyan]")


@app.command(name="daemon")
def daemon_command(
    port: int = typer.Option(DEFAULT_DAEMON_PORT, "--port", "-p", help="Port for the headless STT daemon HTTP server."),
    engine: str = typer.Option("higgs", "--engine", "-e", help="STT engine to keep pre-warmed: 'higgs' or 'whisper'."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Whisper model name (if engine=whisper)."),
    device: str = typer.Option("cuda", "--device", "-d", help="Inference device: 'cuda' or 'cpu'."),
    compute_type: str = typer.Option("float16", "--compute-type", help="Quantization type."),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Default language."),
) -> None:
    """Start background headless STT daemon to keep models resident in VRAM for instant transcription."""
    engine_name = "Higgs Audio v3 STT" if engine.lower() == "higgs" else f"Whisper ({model or 'small.en'})"
    console.print(
        Panel.fit(
            f"[bold green]AutoCut Headless STT Daemon[/bold green]\n\n"
            f"[bold]Pre-warmed Engine:[/] [cyan]{engine_name}[/]\n"
            f"[bold]Device:[/]            [cyan]{device} ({compute_type})[/]\n"
            f"[bold]Listening Address:[/] [bold underline yellow]http://127.0.0.1:{port}[/]\n\n"
            f"[dim]Models remain resident in GPU VRAM for instant zero-latency transcription. Press Ctrl+C to exit.[/dim]",
            title="Resident VRAM Daemon",
            border_style="green",
        )
    )

    daemon = STTDaemon(host="127.0.0.1", port=port)
    try:
        daemon.start(
            engine_type=engine,
            model_name=model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping daemon and releasing VRAM...[/yellow]")
        daemon.stop()
        console.print("[bold green]Daemon stopped successfully.[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
