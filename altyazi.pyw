import faulthandler
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_MAX_BYTES = 2 * 1024 * 1024


def open_log(log_path: Path):
    try:
        if log_path.stat().st_size > LOG_MAX_BYTES:
            os.replace(log_path, log_path.with_name(log_path.stem + ".eski" + log_path.suffix))
    except OSError:
        pass
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} oturum başladı =====\n")
    return log


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))

    log = open_log(here / "altyazi.log")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    faulthandler.enable(file=log, all_threads=True)

    from main import main

    main()
