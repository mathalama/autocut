# autocut

CLI tool for non-destructive screencast editing. This repository currently contains only the approved Stage 0 scaffold; pipeline behavior has not been implemented.

## Install

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Install the approved Stage 1 ASR dependency only in that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install faster-whisper==1.2.1
```

## Stage 1

After opening a new terminal so the ffmpeg PATH change is active, run:

```powershell
.\.venv\Scripts\autocut.exe run recording.mkv --language ru --initial-prompt "Codex, Python, TypeScript, ffmpeg"
```

The default output directory is `work/<recording-name>/` and contains `ingest.json`, `audio.wav`, `speech.json`, and `transcript.json`. Omit `--language` for automatic language detection; write each comparison to a different `--work-dir`.
