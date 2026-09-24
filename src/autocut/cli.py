"""Typer commands that compose only the approved Stage 1 pipeline."""

import json
from pathlib import Path
from typing import Optional

import typer

from autocut.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL
from autocut.models import SpeechIntervals
from autocut.stages.asr import transcribe as transcribe_audio
from autocut.stages.ingest import extract_audio, probe_media
from autocut.stages.vad import detect_speech

app = typer.Typer(help="Create Stage 1 audio and transcript artifacts for a screencast.")


def _default_work_dir(video: Path) -> Path:
    return Path("work") / video.stem


def _ingest(video: Path, work_dir: Path, audio: Optional[Path]) -> Path:
    """Create normalized audio and media metadata in one work directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    media = probe_media(video)
    (work_dir / "ingest.json").write_text(media.model_dump_json(indent=2), encoding="utf-8")
    return extract_audio(audio or video, work_dir / "audio.wav")


@app.command()
def ingest(
    video: Path = typer.Argument(..., exists=True, readable=True),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    audio: Optional[Path] = typer.Option(None, "--audio", exists=True, readable=True),
) -> None:
    """Probe VIDEO and extract its (or --audio's) normalized WAV."""
    output = _ingest(video, work_dir or _default_work_dir(video), audio)
    typer.echo(output)


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
    initial_prompt: Optional[str] = typer.Option(None, "--initial-prompt"),
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
def run(
    video: Path = typer.Argument(..., exists=True, readable=True),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    audio: Optional[Path] = typer.Option(None, "--audio", exists=True, readable=True),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device"),
    compute_type: str = typer.Option(DEFAULT_COMPUTE_TYPE, "--compute-type"),
    language: Optional[str] = typer.Option(None, "--language"),
    initial_prompt: Optional[str] = typer.Option(None, "--initial-prompt"),
) -> None:
    """Run ingest, bundled-ONNX VAD, and timestamped transcription."""
    output_dir = work_dir or _default_work_dir(video)
    audio_path = _ingest(video, output_dir, audio)
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
    typer.echo(json.dumps({"work_dir": str(output_dir), "segments": len(result.segments)}))


def main() -> None:
    """Invoke the CLI entry point installed by the package metadata."""
    app()
