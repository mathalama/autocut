@echo off
chcp 65001 >nul
title AutoCut - Real-Time Live Translation (RU -> EN)

echo ========================================================
echo   AutoCut Live Translation (Russian -^> English)
echo ========================================================
echo.

set URL=http://localhost:8765/?theme=standard^&size=28
powershell -Command "Set-Clipboard -Value '%URL%'" 2>nul

echo [INFO] OBS Browser Source URL copied to clipboard:
echo        %URL%
echo.
echo [TIP]  Speak Russian - English subtitles appear on screen!
echo [TIP]  Press F9 anytime to MUTE / RESUME subtitles!
echo.

if exist ".venv\Scripts\autocut.exe" (
    .\.venv\Scripts\autocut.exe live --translate -l ru -m small
) else (
    echo [ERROR] Virtual environment not found.
    pause
)
pause
