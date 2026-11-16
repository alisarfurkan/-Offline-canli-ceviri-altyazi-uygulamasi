from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_done = False


def register_nvidia_dll_dirs() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return
    for nvidia_dir in spec.submodule_search_locations:
        for bin_dir in Path(nvidia_dir).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    _done = True
