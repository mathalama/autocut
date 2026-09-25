@echo off
chcp 65001 >nul
title AutoCut - Build Standalone EXE

echo ========================================================
echo   AutoCut Standalone EXE Builder
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe build_exe.py
pause
