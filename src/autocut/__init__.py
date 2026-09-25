"""Real-time AI subtitles and captions engine."""

import os
import sys

# Ensure CUDA 12 DLLs from pip wheels (nvidia-cublas, nvidia-cudnn) are discovered on Windows
if sys.platform == "win32":
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    nvidia_dir = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_dir):
        for root, dirs, files in os.walk(nvidia_dir):
            if any(f.endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
                if root not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")

__version__ = "0.1.0"
