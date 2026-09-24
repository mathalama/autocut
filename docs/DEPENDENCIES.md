# Dependency research

Research date: 2026-09-25. Target: Windows x64, CPython 3.10.11 installed in the development environment. Versions are exact pins for repeatability. A package is included only if it has a PyPI distribution; “Windows” means that it can be installed on Windows (a universal `py3-none-any` wheel is acceptable).

| Component | Decision and pinned version | License | Latest release / activity checked | PyPI and Windows | Notes |
|---|---|---|---|---|---|
| ASR | `faster-whisper==1.2.1` | MIT | PyPI release 2025-10-31; upstream repository active | Yes; universal wheel, Python >=3.9 | Approved. Uses CTranslate2; model files download separately. |
| Timing refinement | `stable-ts==2.19.1` | MIT | PyPI release 2025-08-16; upstream marked archived | Yes; source distribution only | Do **not** add by default. Archived and brings PyTorch/Whisper; retain only as a comparison option if faster-whisper word timing proves insufficient. |
| Timing refinement alternative | `whisperx==3.8.6` | BSD-2-Clause | PyPI release 2026-05-25 | Yes; universal wheel, Python >=3.10,<3.14 | Do **not** add by default. It has substantial PyTorch/alignment dependencies. Consider only after a measured timing failure. |
| VAD | `silero-vad==6.2.3` | MIT | PyPI/GitHub release 2026-09-23 | Yes; Windows wheel, Python >=3.8 | Approved for Stage 1. It depends on PyTorch, so belongs to the `pipeline` extra rather than base install. |
| Silence comparison | `auto-editor==29.3.1` | Unlicense | PyPI release 2025-11-04; documentation current | Yes; universal wheel, Python >=3.9 | Optional comparison tool only. Its documented v2/v3 timeline formats can express selected source ranges and it can render v3 files, so it can accept an externally generated timeline. We will nevertheless use ffmpeg for the MVP renderer unless a Stage 2 compatibility test proves its timeline output meets all requirements. |
| Subtitles | `pysubs2==1.8.0` | MIT | Release 2024-12-24 | Yes; universal wheel, Python >=3.9 | Approved. `1.9.0` requires Python >=3.12 and is incompatible with the current CPython 3.10 environment, hence the pin. |
| Timeline export | `opentimelineio==0.18.1` | Apache-2.0 | PyPI release 2025-11-09 | Yes; Windows wheel, Python >3.9 | Approved. OTIO must be supplemented/validated for FCPXML capability during Stage 4. |
| Renderer/probe | `ffmpeg` external executable | LGPL/GPL build-dependent | System binary not found on PATH during Stage 0 | N/A | Approved only as a subprocess, never linked. Install a known Windows build before Stage 1; record build/version in run reports. |
| CLI | `typer==0.27.2` | MIT | PyPI release 2026-08-28 | Yes; universal wheel, Python >=3.10 | Approved. Base dependency. |
| Schemas | `pydantic==2.13.5` | MIT | PyPI release 2026-08-28 | Yes; universal wheel, Python >=3.9 | Approved. Base dependency. |
| Loudness normalization | ffmpeg `loudnorm` filter | Inherits selected ffmpeg build | See ffmpeg row | N/A | No Python dependency. Verify filter availability when ffmpeg is installed. |

## Source checks

- [faster-whisper on PyPI](https://pypi.org/project/faster-whisper/) and its [MIT-licensed upstream repository](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD releases](https://github.com/snakers4/silero-vad/releases) and [PyPI package](https://pypi.org/project/silero-vad/)
- [stable-ts repository](https://github.com/jianfch/stable-ts) and [PyPI package](https://pypi.org/project/stable-ts/)
- [WhisperX on PyPI](https://pypi.org/project/whisperx/)
- [auto-editor v2 format](https://auto-editor.com/docs/v2), [v3 import/render documentation](https://auto-editor.com/docs/v3), and [range syntax](https://auto-editor.com/docs/range-syntax)
- [pysubs2 documentation and MIT license](https://github.com/tkarabela/pysubs2/blob/master/docs/index.rst)
- [OpenTimelineIO on PyPI](https://pypi.org/project/opentimelineio/)
- [Typer license](https://github.com/fastapi/typer/blob/master/LICENSE) and [Pydantic license](https://github.com/pydantic/pydantic/blob/main/LICENSE)

## Outcome

No AGPL/GPL source will be copied. `ffmpeg` is confined to a subprocess boundary. Heavy packages are deferred to the `pipeline` extra, so `pip install -e .` remains a lightweight Stage 0 check. No package is rejected outright, but both timing-refinement candidates are unsuitable as default MVP dependencies for the reasons above.
