# AutoCut: Real-Time AI Subtitles, Live Translation and Video Silence Cutter

AutoCut is a high-performance local audio and video toolkit powered by faster-whisper, NVIDIA CUDA acceleration, and FFmpeg.

It provides two core systems:
1. **Live Subtitles and Translation Engine**: Real-time caption generation and instant spoken translation with ultra-low latency (~150-250 ms) designed for OBS Studio, Twitch, YouTube, and live screencasts.
2. **Video Silence Cutter**: Automated video editing engine that removes dead air, breath pauses, and silence gaps from recordings with frame-accurate precision.

---

## Table of Contents

- [Quick Start (Windows 1-Click)](#quick-start-windows-1-click)
- [Architecture and Pipeline](#architecture-and-pipeline)
- [Feature 1: Real-Time Live Subtitles (autocut live)](#feature-1-real-time-live-subtitles-autocut-live)
- [Feature 2: Real-Time Live Translation (--translate)](#feature-2-real-time-live-translation---translate)
- [Feature 3: Global Hotkey Pause (Mute / Resume)](#feature-3-global-hotkey-pause-mute--resume)
- [Feature 4: OBS Browser Source Overlay and Styling](#feature-4-obs-browser-source-overlay-and-styling)
- [Feature 5: Video Silence Cutter (autocut cut)](#feature-5-video-silence-cutter-autocut-cut)
- [Feature 6: Audio Input Device Management (autocut devices)](#feature-6-audio-input-device-management-autocut-devices)
- [CLI Reference](#cli-reference)
- [Performance Tuning and Troubleshooting](#performance-tuning-and-troubleshooting)
- [License](#license)

---

## Quick Start (Windows 1-Click)

The repository root includes pre-configured batch scripts that start the engine and automatically copy the OBS Browser Source URL directly to your Windows clipboard:

| Script | Purpose |
|---|---|
| **run_live.bat** | Starts live transcription (English default, `small.en` model). Copies overlay URL to clipboard. |
| **run_live_translate.bat** | Starts real-time translation: speak any language (e.g., Russian), and English subtitles render live on screen. |

Double-click the script, paste the URL into your OBS Browser Source, and your broadcast subtitles are ready immediately.

---

## Architecture and Pipeline

```
 Microphone (sounddevice / 16 kHz)
       │  [Pre-allocated circular buffer, 50 ms chunks]
       ▼
 Dynamic Noise Floor VAD (vad.py)
       │  [Rolling 25th percentile baseline tracking]
       ├──(Silence / Fan Hum)──> [0% GPU load, inference skipped]
       │
       ▼ (Audible voice detected)
 faster-whisper + CTranslate2 (CUDA FP16)
       │  [Preloaded DLL handles: cublas64_12, 80-120 ms inference]
       ├─────────────────────────────────┐
       ▼                                 ▼
 Local WebSocket / HTTP Server      Session Recorder (captions.srt)
 (:8765, non-blocking broadcast)     (Millisecond-accurate timecodes)
       │
       ▼
 OBS Browser Source Overlay
 (2-line rolling subtitles, high-contrast container, vanilla CSS)
```

1. **Zero-Allocation Audio Ingest**: `AudioStreamer` utilizes a pre-allocated `CircularAudioBuffer`. Memory is locked at initialization, preventing garbage collector pauses during live streaming.
2. **Dynamic Noise Floor Tracking**: `vad.py` maintains a rolling energy history across recent 100 ms audio frames and continuously tracks the 25th percentile. Background fan noise and microphone hiss are treated as baseline floor rather than speech.
3. **Hardware Acceleration**: Windows CUDA libraries (`cublas64_12.dll`, `cublasLt64_12.dll`, and `nvrtc`) are preloaded into process memory via ctypes before CTranslate2 initialization, ensuring immediate GPU offloading without DLL load failures.
4. **Low-Latency Streaming**: Intermediate tokens are dispatched every 150 ms while speaking. When a 0.6-second pause occurs, the phrase is finalized, committed to `captions.srt`, and rolled onto the upper line of the overlay.

---

## Feature 1: Real-Time Live Subtitles (autocut live)

The core live engine transcribes spoken audio into text with low latency.

```powershell
# Default English setup (CUDA, small.en model, 150 ms step)
.\.venv\Scripts\autocut.exe live

# Russian speech recognition using multilingual model
.\.venv\Scripts\autocut.exe live -l ru -m small

# Custom noise gate for loud or noisy microphone hardware
.\.venv\Scripts\autocut.exe live --energy-threshold 0.025
```

### Runtime Behavior:
- The terminal displays live progress tokens (`interim words`) and final sentences (`[Final]`).
- Subtitles are streamed to the OBS browser overlay and recorded to `captions.srt` with synchronized timestamps.

---

## Feature 2: Real-Time Live Translation (--translate)

Allows streamers to broadcast to international audiences:

- **How it works**: Speak in your native language (Russian, Spanish, German, Japanese, etc.), and the Whisper decoder outputs fluent English subtitles live in OBS.
- The engine automatically resolves `.en` models to multilingual variants (`small` or `base`) when translation is requested.

```powershell
# Speak Russian, render English subtitles on stream:
.\.venv\Scripts\autocut.exe live --translate -l ru -m small

# Alternatively, use the 1-click Windows launcher:
run_live_translate.bat
```

---

## Feature 3: Global Hotkey Pause (Mute / Resume)

Streamers often need to temporarily mute subtitles during private conversations, phone calls, or breaks.

- **Default Key**: `F9`
- **Behavior**:
  - The hotkey is hooked globally using the Windows keyboard API. It functions inside full-screen games and third-party apps without needing to switch windows.
  - First press: Captions are cleared from OBS, audio inference pauses, and the terminal displays `[PAUSED]`.
  - Second press: Captions resume immediately with `[RESUMED]`.
- **Custom Hotkey Binding**:
  ```powershell
  .\.venv\Scripts\autocut.exe live --hotkey pause
  .\.venv\Scripts\autocut.exe live --hotkey f10
  ```

---

## Feature 4: OBS Browser Source Overlay and Styling

The frontend uses a 2-line rolling layout:
- **Line 1 (Upper)**: The previously finalized sentence, rendered at slight opacity for reading context.
- **Line 2 (Lower)**: Current live sentence streaming in real time.

### OBS Studio Setup:
1. In OBS, click `+` under **Sources** and select **Browser**.
2. Paste the target URL into the **URL** field.
3. Set **Width** to `1920` and **Height** to `1080`.
4. Check **Shutdown source when not visible** and **Refresh browser when scene becomes active**.

### URL Parameters Reference:

All visual parameters are configured via URL query parameters without restarting the server:

| Style Goal | URL |
|---|---|
| **Standard (Default)**<br>Crisp white text on dark plate (rgba 0, 0, 0, 0.78) | `http://localhost:8765/?theme=standard&size=28` |
| **Black on White**<br>High-contrast dark text on bright plate | `http://localhost:8765/?theme=black&size=28` |
| **Outline Mode**<br>White text with deep black stroke, transparent background | `http://localhost:8765/?theme=outline&size=32` |
| **Yellow Subtitles**<br>Classic broadcast cinema yellow on dark background | `http://localhost:8765/?theme=yellow&size=28` |
| **Top Screen Position** | `http://localhost:8765/?theme=standard&pos=top` |
| **Center Screen Position** | `http://localhost:8765/?theme=standard&pos=center` |
| **Custom Size** | `http://localhost:8765/?size=38` |
| **Fully Customized**<br>(custom text color, background, font size, alignment) | `http://localhost:8765/?color=yellow&bg=rgba(0,0,0,0.85)&size=32&align=center` |

---

## Feature 5: Video Silence Cutter (autocut cut)

Automated editing tool for YouTube videos, podcasts, and screencasts. It detects silence and dead air, cuts unnecessary pauses, and splices the media with frame accuracy.

```powershell
# Basic silence removal:
.\.venv\Scripts\autocut.exe cut raw_recording.mp4 -o trimmed_video.mp4
```

### Advanced Cutting Parameters:

```powershell
# Cut gaps longer than 0.5s, keep 120 ms audio padding, and export synchronized SRT:
.\.venv\Scripts\autocut.exe cut input.mp4 -o output.mp4 --pause-threshold 0.5 --margin 0.12 --output-srt output.srt
```

### Technical Highlights:
- **Sample-Accurate Splicing**: Generates an FFmpeg `filter_complex` pipeline using paired `trim`/`atrim` and `concat` operations. Audio and video tracks stay perfectly in sync over long files.
- **Speech Padding Margin**: Adds configurable margins (`margin=0.15s` by default) around every speech interval to prevent clipped syllables.
- **Synchronized Subtitles**: If `--output-srt` is supplied, speech timestamps are automatically remapped to match the newly edited timeline.
- **Detailed Summary**: Prints duration before and after, time saved, reduction percentage, and segment count.

---

## Feature 6: Audio Input Device Management (autocut devices)

To inspect connected audio interfaces (USB microphones, interfaces, headsets):

```powershell
.\.venv\Scripts\autocut.exe devices
```

Example output:
```
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Index ┃ Device Name                    ┃ Channels ┃ SampleRate ┃ Default ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1     │ Microphone (HyperX QuadCast)   │ 2        │ 48000 Hz   │ YES     │
│ 3     │ Microphone (HD Pro Webcam C920)│ 2        │ 16000 Hz   │         │
└───────┴────────────────────────────────┴──────────┴────────────┴─────────┘
```

Bind the engine to a specific input device index:
```powershell
.\.venv\Scripts\autocut.exe live --device-index 1
```

---

## CLI Reference

### Command: autocut live
```
Usage: autocut live [OPTIONS]

Options:
  -m, --model TEXT            Whisper model name: 'small.en', 'tiny.en', 'base', 'small', 'medium' [default: small.en]
  -d, --device TEXT           Inference device: 'cuda' or 'cpu' [default: cuda]
  --compute-type TEXT         Quantization: 'float16', 'int8', 'float32' [default: float16]
  -l, --language TEXT         Language code: 'en', 'ru', 'auto', etc. [default: en]
  -T, --translate             Enable real-time spoken translation to English
  -k, --hotkey TEXT           Global hotkey to toggle pause/mute [default: f9]
  --energy-threshold FLOAT    VAD noise floor threshold (0.015-0.035 for noisy mics) [default: 0.015]
  -p, --port INTEGER          Server port for HTTP and WebSocket [default: 8765]
  --device-index INTEGER      Microphone index from 'autocut devices'
  -o, --output-srt PATH       Path to record subtitle session [default: captions.srt]
  -t, --theme TEXT            Overlay theme: 'standard', 'black', 'outline', 'yellow' [default: standard]
  -s, --size INTEGER          Overlay font size in pixels [default: 28]
  --help                      Show this message and exit.
```

### Command: autocut cut
```
Usage: autocut cut [OPTIONS] INPUT_VIDEO

Arguments:
  INPUT_VIDEO                 Path to input video or audio file (MP4, MKV, MOV, WAV) [required]

Options:
  -o, --output PATH           Path to export cut file [default: {name}_cut.mp4]
  -m, --model TEXT            Whisper model for detection [default: base]
  -d, --device TEXT           'cuda' or 'cpu' [default: cuda]
  -p, --pause-threshold FLOAT Minimum silence duration in seconds to cut out [default: 0.6]
  --margin FLOAT              Audio padding margin around speech in seconds [default: 0.15]
  -l, --language TEXT         Spoken language code ('en', 'ru', etc.)
  --output-srt PATH           Path to export synchronized subtitles
  --help                      Show this message and exit.
```

---

## Performance Tuning and Troubleshooting

### 1. Verifying CUDA Acceleration
When starting `autocut live`, check the status banner:
```
Device: cuda (float16)
```
If CUDA libraries are missing or an unsupported GPU is present, the engine logs a warning and automatically falls back to CPU quantization (`cpu (int8)`).

### 2. High Microphone Noise or Fan Hum
If random tokens or phantom periods appear during silence:
- Increase the noise threshold:
  ```powershell
  .\.venv\Scripts\autocut.exe live --energy-threshold 0.025
  ```

### 3. Whisper Model Recommendations
- **small.en** (Default for English): Best speed-to-accuracy ratio on GPU (~100 ms per chunk).
- **base** (Multilingual): Fastest multilingual model (~60 ms on GPU). Ideal for video silence cutting (`autocut cut`).
- **small** (Multilingual): Recommended for Russian recognition and real-time translation (`--translate`).
- **large-v3-turbo**: Maximum vocabulary capacity for technical terms and whisper audio (requires ~3 GB VRAM).

---

## License

MIT License. Free for personal, educational, and commercial streaming, recording, and broadcasting.
