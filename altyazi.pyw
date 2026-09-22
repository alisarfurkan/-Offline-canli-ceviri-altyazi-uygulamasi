import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
os.chdir(here)
sys.path.insert(0, str(here))

log = open(here / "altyazi.log", "w", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = log
if sys.stderr is None:
    sys.stderr = log

from main import main

main()
