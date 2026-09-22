from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from config import Settings
from overlay_ui import SubtitleOverlay
from pipeline import Pipeline
from tray_app import TrayApp


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    settings = Settings.load()

    overlay = SubtitleOverlay()
    overlay.set_locked(settings.locked)
    overlay.show()

    state = {"pipeline": None}

    def on_result(original: str, translated: str) -> None:
        overlay.set_subtitle(original, translated)

    def on_status(message: str) -> None:
        overlay.set_text(message)

    def start_pipeline() -> None:
        pipeline = Pipeline(settings, on_result=on_result, on_status=on_status)
        pipeline.start()
        state["pipeline"] = pipeline

    def stop_pipeline() -> None:
        pipeline: Pipeline | None = state["pipeline"]
        if pipeline is not None:
            pipeline.stop()
            state["pipeline"] = None
        overlay.set_text("Durduruldu")

    def toggle_lock(locked: bool) -> None:
        overlay.set_locked(locked)

    def on_settings_changed() -> None:
        pass

    tray = TrayApp(
        settings=settings,
        on_start=start_pipeline,
        on_stop=stop_pipeline,
        on_toggle_lock=toggle_lock,
        on_settings_changed=on_settings_changed,
    )
    tray.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
