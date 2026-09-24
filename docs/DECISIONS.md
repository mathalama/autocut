# Decisions

## 2026-09-25 — Stage 0 dependency layout

The base installation contains only the CLI and schema libraries. Stage-specific ASR/VAD/export packages are exact-pinned optional extras so the required editable-install check does not download PyTorch or models before Stage 1 is approved.

## 2026-09-25 — Rendering boundary

The intended MVP renderer remains ffmpeg via `subprocess`. auto-editor's v2/v3 formats are capable of carrying externally selected ranges, but it remains a comparison/contingency dependency until a Stage 2 compatibility test validates the output.

## 2026-09-25 — Stage 1 VAD and hardware

`silero-vad==6.2.3` was not installed: its resolver plan includes `torch>=1.12.0`. `faster-whisper==1.2.1` already packages an ONNX Silero VAD and exposes `faster_whisper.vad.get_speech_timestamps`, which returns speech intervals as `{start, end}` sample positions at 16 kHz. Stage 1 will use that function and serialize seconds to `speech.json`; this produces the same VAD intervals used by faster-whisper when supplied the same audio and `VadOptions`.

The machine has an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GiB VRAM, driver 610.47. `ctranslate2==4.8.2` reports one CUDA device and supports `float16`; Stage 1 defaults to `large-v3-turbo`, `device="cuda"`, `compute_type="float16"`. No CUDA toolkit directory is required by CTranslate2 in the observed environment. Minimum GPU requirement for this default is an NVIDIA CUDA-capable GPU with 8 GiB VRAM, an installed driver compatible with CUDA 12/cuDNN 9, and adequate disk for the model/cache. CPU fallback: `device="cpu"`, `compute_type="int8"`, model `small` (or `turbo` if memory allows), 16 GiB RAM recommended; processing will be materially slower.

Only CPython 3.10.11 is installed (`py -0p`); Python 3.12 is unavailable. The project-local `.venv` therefore uses 3.10.11 and contains all new packages; the global environment is untouched after this decision.

ffmpeg 9.0.2 (Gyan shared build) was installed with winget for ingest. winget added its `bin` directory to the user PATH; open a new terminal before invoking `autocut` so `ffmpeg` and `ffprobe` resolve by name.

## 2026-09-25 — Stage 1 accepted baseline

The approved default is `large-v3-turbo` with `language="ru"` and an optional, user-supplied `initial_prompt`; automatic language detection remains available for comparison. No three-run benchmark was executed because no real recording path was supplied (the supplied value was the literal placeholder `<путь>`). Consequently, RTF, peak VRAM, and the list of incorrectly recognized English/technical words are recorded as **not measured**, rather than fabricated. The CLI accepts separate `--work-dir` values for the three required variants so their `transcript.json` artifacts and future measurements do not overwrite each other.
