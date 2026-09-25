# AutoCut: AI Audio Recording, Subtitles and Real-Time OBS Streaming

AutoCut is a local audio, subtitle, and video toolkit powered by NVIDIA CUDA, faster-whisper, Boson AI Higgs Audio v3 STT, and FFmpeg.

It provides four modular capabilities:
1. **Studio Subtitle Recorder (`autocut record`) - Main Feature**:
   Press record, speak naturally without streaming latency pressure, and press stop (Enter or F9) to generate high-accuracy SRT and TXT subtitles with automatic clipboard copying.
2. **Real-Time Live Subtitles (`autocut live`)**:
   Single-pass final engine designed for OBS Studio, Twitch, YouTube, and screencasts. Outputs clean, finalized captions with zero flicker.
3. **Media File Transcription (`autocut transcribe`)**:
   Direct transcription of existing video and audio files (MP4, MKV, MOV, WAV, MP3) into synchronized SRT and TXT subtitle files.
4. **Video Silence Cutter (`autocut cut`)**:
   Automated dead-air and pause removal with frame-accurate FFmpeg splicing.

---

## Quick Start (Windows 1-Click)

The repository root includes pre-configured batch scripts:

| Script | Purpose |
|---|---|
| `install.bat` | One-click setup: creates Python virtual environment and installs dependencies with PyTorch CUDA. |
| `run_record.bat` | Starts studio recorder mode using Higgs Audio v3 STT / Whisper. Press Enter or F9 to stop and automatically copy subtitles to clipboard. |
| `run_live.bat` | Starts live subtitle stream for OBS Studio. Automatically copies overlay URL to clipboard. |
| `run_live_translate.bat` | Starts real-time translation: spoken non-English speech renders as English subtitles in OBS. |

---

## Architecture

```
 Microphone (sounddevice / 16 kHz)
        |
        +---> Mode 1: autocut record (Studio Recording - Main)
        |       |
        |       +---> In-memory lossless audio stream
        |       +---> RollingTranscriber: pre-emptive VAD phrase transcription on GPU
        |       +---> Instant Stop (Enter / F9): final subtitles assembled in < 0.3s
        |             |
        |             +---> captions.srt (timestamped for video editing)
        |             +---> captions.txt (clean text)
        |             +---> Windows Clipboard (ready to paste)
        |
        +---> Mode 2: autocut live (OBS Live Streaming)
        |       |
        |       +---> Adaptive noise-floor VAD
        |       +---> Single-Pass inference (finalized text only)
        |       +---> WebSocket / HTTP (:8765) -> OBS Browser Source
        |
        +---> Mode 3: autocut daemon (Headless Resident VRAM Service)
                |
                +---> Pre-warms Higgs Audio v3 STT or Whisper in GPU memory
                +---> HTTP API (:8766): /transcribe, /transcribe_chunk, /switch_model
                +---> Zero cold-start latency for all client calls
```

---

## Main Feature: Studio Subtitle Recorder (`autocut record`)

Ideal for content creators, video editors (Premiere, DaVinci, CapCut), voice memos, and podcasts. Record speech without latency constraints and let the neural network transcribe the complete segment with full context.

Thanks to the **RollingTranscriber** pipeline, completed phrases are pre-emptively recognized in the background while you are speaking. When you stop recording, results appear almost instantaneously.

```powershell
# 1-Click launch via Windows batch file:
.\run_record.bat

# Standard CLI launch (Whisper base, CUDA):
.\.venv\Scripts\autocut.exe record

# Using Higgs Audio v3 STT (LLM-decoder backend):
.\.venv\Scripts\autocut.exe record --engine higgs

# Specifying language and custom model:
.\.venv\Scripts\autocut.exe record -l en -m small.en

# Specifying custom output filename:
.\.venv\Scripts\autocut.exe record -o my_subtitles.srt
```

### Workflow:
1. Recording starts with an active elapsed timer in the terminal.
2. Press **ENTER** in the console or press **F9** (global hotkey across all applications) to stop recording.
3. The background worker transcribes completed phrases on-the-fly.
4. The STT engine finalizes:
   - `captions.srt` (timecoded subtitles)
   - `captions.txt` (raw transcript)
   - Automatically copied to the Windows clipboard for instant pasting.

---

## Streaming Feature: Real-Time Subtitles for OBS (`autocut live`)

Designed for live streams and voice calls. The **Single-Pass Final Engine** eliminates interim draft flickering. As soon as a speaker pauses (0.4s threshold), the completed sentence is transcribed once and emitted directly as a final subtitle.

```powershell
# Default fast Whisper setup (CUDA, small.en model):
.\.venv\Scripts\autocut.exe live

# Using Higgs Audio v3 STT:
.\.venv\Scripts\autocut.exe live --engine higgs

# Live translation into English:
.\.venv\Scripts\autocut.exe live --translate -l ru -m small

# Custom noise gate for loud microphones or background noise:
.\.venv\Scripts\autocut.exe live --energy-threshold 0.025
```

### Controls During Live Streaming:
- **Global Hotkey (`F9`)**: Toggles subtitle mute/pause instantly without switching windows.
- **OBS Integration**: Add a **Browser Source** pointing to `http://localhost:8765/?theme=standard&size=28` with 1920x1080 resolution.

### Overlay Themes:
- Standard (white text, dark plate): `?theme=standard&size=28`
- Cinema Yellow: `?theme=yellow&size=28`
- Outline (transparent background): `?theme=outline&size=32`
- Black on White: `?theme=black&size=28`
- Position customization: `&pos=top` or `&pos=center`

---

## Feature 3: Resident VRAM Daemon (`autocut daemon`)

Starts a background local HTTP service that keeps heavy models (such as Higgs Audio v3 STT or Whisper large) resident in GPU memory to eliminate model reloading delays:

```powershell
# Start daemon with Higgs Audio v3 STT:
.\.venv\Scripts\autocut.exe daemon --engine higgs --port 8766

# Start daemon with Whisper:
.\.venv\Scripts\autocut.exe daemon --engine whisper -m small.en
```

---

## Feature 4: File Transcription (`autocut transcribe`)

Transcribe pre-recorded media files (MP4, MKV, MOV, WAV, MP3) into synchronized subtitles:

```powershell
# Transcribe video to subtitles:
.\.venv\Scripts\autocut.exe transcribe input.mp4 -o output.srt

# Transcribe with Higgs Audio v3 STT:
.\.venv\Scripts\autocut.exe transcribe voice_memo.wav --engine higgs
```

---

## Feature 5: Video Silence Cutter (`autocut cut`)

Automatically detects silence intervals and splices out dead air with sample accuracy:

```powershell
# Basic silence removal:
.\.venv\Scripts\autocut.exe cut raw_video.mp4 -o edited_video.mp4

# Remove pauses longer than 0.5s with 120ms speech padding and export remapped SRT:
.\.venv\Scripts\autocut.exe cut raw.mp4 -o out.mp4 --pause-threshold 0.5 --margin 0.12 --output-srt out.srt
```

---

## Audio Input Device Management (`autocut devices`)

List available audio interfaces:

```powershell
.\.venv\Scripts\autocut.exe devices
```

Specify a device index:
```powershell
.\.venv\Scripts\autocut.exe record --device-index 1
.\.venv\Scripts\autocut.exe live --device-index 1
```

---

## Supported STT Engines

1. **faster-whisper (CTranslate2)**:
   - Models: `tiny`, `base`, `small`, `medium`, `large-v3-turbo`.
   - Low VRAM footprint (500 MB - 2 GB) with CUDA float16 acceleration.
2. **Higgs Audio v3 STT (`bosonai/higgs-audio-v3-stt`)**:
   - Whisper-Large-v3 acoustic encoder coupled with a Qwen3-based language model decoder.
   - High contextual understanding and punctuation accuracy.
   - Activated via the `--engine higgs` flag.

---

## CLI Reference Summary

### autocut record
```
Options:
  -o, --output-srt PATH       Path to export SRT subtitles [default: captions.srt]
  -e, --engine TEXT           STT engine: 'higgs' or 'whisper' [default: higgs]
  -m, --model TEXT            Whisper model name [default: large-v3-turbo]
  -d, --device TEXT           Inference device: 'cuda' or 'cpu' [default: cuda]
  --compute-type TEXT         Quantization type [default: float16]
  -l, --language TEXT         Language code ('auto', 'en', 'ru', etc.) [default: auto]
  -k, --hotkey TEXT           Hotkey to stop recording [default: f9]
  --device-index INTEGER      Microphone index from 'autocut devices'
  -w, --save-wav PATH         Optional path to save recorded audio WAV
  --clipboard / --no-clipboard Automatically copy text to clipboard [default: True]
```

### autocut live
```
Options:
  -m, --model TEXT            Model name [default: small.en]
  -d, --device TEXT           Inference device [default: cuda]
  --compute-type TEXT         Quantization [default: float16]
  -l, --language TEXT         Language code [default: en]
  -T, --translate             Enable real-time spoken translation to English
  -k, --hotkey TEXT           Global hotkey to toggle pause/mute [default: f9]
  --energy-threshold FLOAT    Noise gate threshold [default: 0.015]
  -p, --port INTEGER          Server port [default: 8765]
  -o, --output-srt PATH       Path to record subtitle session [default: captions.srt]
  -t, --theme TEXT            Theme: 'standard', 'black', 'outline', 'yellow'
  -s, --size INTEGER          Font size in pixels [default: 28]
  -e, --engine TEXT           STT engine: 'whisper' or 'higgs' [default: whisper]
```

### autocut daemon
```
Options:
  -p, --port INTEGER          Daemon HTTP port [default: 8766]
  -e, --engine TEXT           Pre-warmed STT engine: 'higgs' or 'whisper' [default: higgs]
  -m, --model TEXT            Model name
  -d, --device TEXT           Inference device [default: cuda]
  --compute-type TEXT         Quantization [default: float16]
  -l, --language TEXT         Default language code
```

### autocut transcribe
```
Arguments:
  INPUT_FILE                  Path to audio or video file (MP4, MKV, WAV, MP3) [required]

Options:
  -o, --output-srt PATH       Path to export SRT subtitles
  -e, --engine TEXT           STT engine: 'higgs' or 'whisper' [default: higgs]
  -m, --model TEXT            Whisper model name [default: large-v3-turbo]
  -d, --device TEXT           Inference device [default: cuda]
  --compute-type TEXT         Quantization [default: float16]
  -l, --language TEXT         Language code [default: auto]
  --clipboard / --no-clipboard Copy text to Windows clipboard
```

### autocut cut
```
Arguments:
  INPUT_VIDEO                 Path to input video or audio file [required]

Options:
  -o, --output PATH           Path to export trimmed video
  -m, --model TEXT            Whisper model name [default: base]
  -d, --device TEXT           Inference device [default: cuda]
  -p, --pause-threshold FLOAT Minimum silence duration in seconds [default: 0.6]
  --margin FLOAT              Audio padding margin in seconds [default: 0.15]
  -l, --language TEXT         Spoken language code
  --output-srt PATH           Export synchronized subtitles
```

### autocut devices
```
Usage: autocut devices
Prints all active audio input interfaces, channels, sample rates, and default status.
```

---

## License

MIT License. Free for personal and commercial broadcasting, recording, and editing.
