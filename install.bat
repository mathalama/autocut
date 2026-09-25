@echo off
chcp 65001 >nul
title AutoCut - Automatic Installer

echo ========================================================
echo   AutoCut Setup and Environment Initializer
echo ========================================================
echo.

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Using 'uv' fast package manager...
    uv venv .venv
    echo [INFO] Installing PyTorch with CUDA acceleration...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo [INFO] Installing AutoCut dependencies...
    uv pip install -e .
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [OK] Using Python from system PATH...
        python -m venv .venv
        echo [INFO] Upgrading pip...
        .\.venv\Scripts\python.exe -m pip install --upgrade pip
        echo [INFO] Installing PyTorch with CUDA acceleration...
        .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        echo [INFO] Installing AutoCut dependencies...
        .\.venv\Scripts\python.exe -m pip install -e .
    ) else (
        echo [ERROR] Python was not found on your system.
        echo Please install Python 3.10+ from: https://www.python.org/downloads/
        echo IMPORTANT: Check the box "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

echo.
echo ========================================================
echo   [OK] Setup complete! You can now launch:
echo         - run_record.bat (Studio recording)
echo         - run_live.bat   (Live streaming for OBS)
echo ========================================================
echo.
pause
