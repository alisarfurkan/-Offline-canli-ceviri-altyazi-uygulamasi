from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False


def register_nvidia_dll_dirs() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return
    for bin_dir in nvidia_dir.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    _done = True
