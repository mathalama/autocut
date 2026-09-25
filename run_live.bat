@echo off
chcp 65001 >nul
title AutoCut - Real-Time Live Subtitles

echo ========================================================
echo   AutoCut Live Subtitles Engine
echo ========================================================
echo.

set URL=http://localhost:8765/?theme=standard^&size=28
powershell -Command "Set-Clipboard -Value '%URL%'" 2>nul

echo [INFO] OBS Browser Source URL copied to clipboard:
echo        %URL%
echo.
echo [TIP]  Press F9 anytime to MUTE / RESUME subtitles!
echo.

if exist ".venv\Scripts\autocut.exe" (
    .\.venv\Scripts\autocut.exe live
) else (
    echo [ERROR] Virtual environment not found. Please run:
    echo        uv venv .venv
    echo        uv pip install -e .
    pause
)
pause
