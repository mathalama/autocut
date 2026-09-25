"""Typer CLI commands for real-time autocut subtitles."""

from pathlib import Path
import time
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from autocut.audio import list_input_devices
from autocut.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_WS_PORT,
)
from autocut.engine import RealtimeSubtitleEngine
from autocut.recorder import CaptionRecorder
from autocut.server import OverlayServer

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
            "★ [bold green]YES[/]" if d["is_default"] else "",
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
    theme: str = typer.Option("glass", "--theme", "-t", help="Overlay theme: 'glass', 'neon', 'cinema', 'minimal'."),
    size: int = typer.Option(28, "--size", "-s", help="Font size in pixels for the overlay."),
) -> None:
    """Start real-time subtitle generation with OBS Browser Source overlay."""
    actual_port = ws_port if ws_port else port
    obs_url = f"http://localhost:{actual_port}/?theme={theme}&size={size}"

    console.print(
        Panel.fit(
            f"[bold green]AutoCut Real-Time Subtitles[/bold green]\n\n"
            f"[bold]Language:[/] [cyan]{language}[/]   "
            f"[bold]Model:[/] [cyan]{model}[/]   "
            f"[bold]Device:[/] [cyan]{device} ({compute_type})[/]\n"
            f"[bold]OBS Browser Source URL:[/] [bold underline yellow]{obs_url}[/]\n"
            f"[bold]Saving Subtitles to:[/] [cyan]{output_srt}[/]\n\n"
            f"[dim]Press Ctrl+C to stop.[/dim]",
            title="★ Engine Active",
            border_style="cyan",
        )
    )

    recorder = CaptionRecorder(srt_path=output_srt, txt_path=output_srt.with_suffix(".txt") if output_srt else None)
    server = OverlayServer(host="0.0.0.0", port=actual_port)
    server.start()

    session_start = time.time()
    last_caption_time = session_start

    def on_caption(text: str, is_final: bool) -> None:
        nonlocal last_caption_time
        now = time.time()
        start_rel = last_caption_time - session_start
        end_rel = now - session_start

        # Broadcast to OBS Browser Source
        server.broadcast({
            "type": "caption",
            "text": text,
            "is_final": is_final,
        })

        if is_final:
            import sys
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            console.print(f"[bold green]✔ Final:[/] [white]{text}[/]")
            recorder.add_caption(text, start_rel, end_rel)
            last_caption_time = now
        else:
            import sys
            sys.stdout.write(f"\r\033[K  ● {text}")
            sys.stdout.flush()

    def on_clear() -> None:
        server.broadcast({"type": "clear"})

    engine = RealtimeSubtitleEngine(
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
        device_index=device_index,
        energy_threshold=energy_threshold,
        on_caption=on_caption,
        on_clear=on_clear,
    )

    console.print("[dim]Loading model and starting audio listener...[/dim]")
    engine.start()
    console.print("[bold green]✔ Listening to microphone! Speak now...[/bold green]\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live subtitles...[/yellow]")
    finally:
        engine.stop()
        server.stop()
        console.print(f"[bold green]✔ Done. Subtitles saved to [cyan]{output_srt}[/cyan].[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
