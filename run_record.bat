@echo off
chcp 65001 >nul
title AutoCut - High-Accuracy Studio Subtitles (Higgs STT / Whisper)

echo ========================================================
echo   AutoCut Studio Subtitle Recorder (Main Feature)
echo ========================================================
echo.
echo [INFO]  Speak into your microphone.
echo [TIP]   Press F9 or ENTER when finished to STOP and generate subtitles!
echo.

if not exist ".venv\Scripts\autocut.exe" (
    echo [INFO] First time setup detected. Launching automatic installer...
    call install.bat
)

if exist ".venv\Scripts\autocut.exe" (
    .\.venv\Scripts\autocut.exe record --engine higgs
) else (
    echo [ERROR] Installation was cancelled or not completed.
    pause
)
pause
