# Decisions

## 2026-09-25 — Stage 0 dependency layout

The base installation contains only the CLI and schema libraries. Stage-specific ASR/VAD/export packages are exact-pinned optional extras so the required editable-install check does not download PyTorch or models before Stage 1 is approved.

## 2026-09-25 — Rendering boundary

The intended MVP renderer remains ffmpeg via `subprocess`. auto-editor's v2/v3 formats are capable of carrying externally selected ranges, but it remains a comparison/contingency dependency until a Stage 2 compatibility test validates the output.
