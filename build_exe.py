"""Build script to create a standalone AutoCut.exe distribution."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "src" / "autocut" / "static"
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"


def build() -> None:
    print("=" * 60)
    print("  AutoCut Standalone Executable Builder")
    print("=" * 60)

    # Clean previous build artifacts
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            print(f"Cleaning {d.name}...")
            shutil.rmtree(d, ignore_errors=True)

    static_spec = f"{STATIC_DIR};autocut/static"

    pyinstaller_cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name", "AutoCut",
        "--onedir",
        "--console",
        f"--add-data={static_spec}",
        "--hidden-import=faster_whisper",
        "--hidden-import=ctranslate2",
        "--hidden-import=sounddevice",
        "--hidden-import=keyboard",
        "--hidden-import=websockets",
        "--hidden-import=typer",
        "--hidden-import=rich",
        "--hidden-import=torch",
        "--hidden-import=transformers",
        "--hidden-import=accelerate",
        "--hidden-import=autocut",
        "--hidden-import=autocut.stt",
        "--hidden-import=autocut.stt.whisper_engine",
        "--hidden-import=autocut.stt.higgs_engine",
        "--hidden-import=autocut.stt.rolling",
        "--hidden-import=autocut.stt.pool",
        "--hidden-import=autocut.daemon",
        "--noconfirm",
        str(BASE_DIR / "run_app.py"),
    ]

    print("\nRunning PyInstaller...")
    print(" ".join(pyinstaller_cmd[:6]) + " ...")
    subprocess.run(pyinstaller_cmd, check=True)

    exe_path = DIST_DIR / "AutoCut" / "AutoCut.exe"
    if exe_path.exists():
        print("\n" + "=" * 60)
        print("  [SUCCESS] AutoCut.exe build completed successfully!")
        print(f"  Location: {exe_path}")
        print("=" * 60)
    else:
        print("\n[ERROR] AutoCut.exe was not created.")
        sys.exit(1)


if __name__ == "__main__":
    build()
