"""High-performance silence and dead-air cutter for video recordings."""

import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def get_media_duration(file_path: Path) -> float:
    """Extract media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        logger.error("Failed to read duration from %s: %s", file_path, e)
        raise RuntimeError(f"Cannot read media duration for {file_path}") from e


def extract_audio_16k(video_path: Path, output_wav: Path) -> None:
    """Extract mono 16kHz audio from video file for transcription/VAD."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def compute_keep_intervals(
    speech_segments: List[Tuple[float, float]],
    total_duration: float,
    min_pause: float = 0.6,
    margin: float = 0.15,
) -> List[Tuple[float, float]]:
    """Convert raw speech intervals into padded keep segments, removing pauses > min_pause."""
    if not speech_segments:
        return [(0.0, total_duration)]

    # 1. Add padding margin around each speech segment
    padded = []
    for start, end in speech_segments:
        p_start = max(0.0, start - margin)
        p_end = min(total_duration, end + margin)
        padded.append((p_start, p_end))

    # 2. Merge overlapping or closely adjacent intervals
    merged = []
    current_start, current_end = padded[0]
    for start, end in padded[1:]:
        if start <= current_end + min_pause:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    # 3. Filter out micro-segments (< 0.2s)
    clean = [(s, e) for s, e in merged if (e - s) >= 0.2]
    return clean or [(0.0, total_duration)]


def cut_video(
    input_file: Path,
    output_file: Path,
    model_name: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
    pause_threshold: float = 0.6,
    margin: float = 0.15,
    language: Optional[str] = None,
    output_srt: Optional[Path] = None,
    progress_callback: Optional[Any] = None,
) -> dict[str, Any]:
    """Analyze video with VAD/Whisper, cut silent pauses, and export trimmed video with SRT."""
    total_duration = get_media_duration(input_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "extracted_16k.wav"
        if progress_callback:
            progress_callback("Extracting audio track...")
        extract_audio_16k(input_file, tmp_wav)

        if progress_callback:
            progress_callback("Transcribing & analyzing speech intervals...")

        # Lazy import of engine DLL registration
        from autocut.engine import RealtimeSubtitleEngine
        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception:
            logger.warning("CUDA unavailable, falling back to CPU int8 for cutting")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")

        segments_iter, _ = model.transcribe(
            str(tmp_wav),
            language=language,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=int(pause_threshold * 1000),
                speech_pad_ms=int(margin * 1000),
            ),
        )

        speech_intervals = []
        captions_data = []
        for seg in segments_iter:
            if seg.text.strip():
                speech_intervals.append((seg.start, seg.end))
                captions_data.append((seg.start, seg.end, seg.text.strip()))

        keep_intervals = compute_keep_intervals(
            speech_intervals,
            total_duration=total_duration,
            min_pause=pause_threshold,
            margin=margin,
        )

        cut_duration = sum(end - start for start, end in keep_intervals)
        time_saved = max(0.0, total_duration - cut_duration)
        ratio = (time_saved / total_duration * 100) if total_duration > 0 else 0.0

        if progress_callback:
            progress_callback(f"Rendering cut video ({len(keep_intervals)} segments)...")

        # Build FFmpeg filter_complex for exact A/V frame-accurate splicing
        filter_parts = []
        concat_inputs = []
        for idx, (start, end) in enumerate(keep_intervals):
            filter_parts.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{idx}];"
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{idx}]"
            )
            concat_inputs.append(f"[v{idx}][a{idx}]")

        filter_complex = (
            ";".join(filter_parts)
            + f";{''.join(concat_inputs)}concat=n={len(keep_intervals)}:v=1:a=1[outv][outa]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_file),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_file),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # Write synchronized output SRT if requested
        if output_srt:
            # Map original timestamps to new cut timeline
            from autocut.recorder import CaptionRecorder
            recorder = CaptionRecorder(srt_path=output_srt)
            for seg_start, seg_end, text in captions_data:
                # Find corresponding position in cut timeline
                new_start = 0.0
                mapped = False
                for k_start, k_end in keep_intervals:
                    if seg_start < k_start:
                        break
                    if k_start <= seg_start <= k_end:
                        new_start += (seg_start - k_start)
                        mapped = True
                        break
                    new_start += (k_end - k_start)
                if mapped:
                    new_end = new_start + (seg_end - seg_start)
                    recorder.add_caption(text, new_start, new_end)

        return {
            "original_duration": total_duration,
            "cut_duration": cut_duration,
            "time_saved": time_saved,
            "reduction_percent": ratio,
            "num_segments": len(keep_intervals),
        }
