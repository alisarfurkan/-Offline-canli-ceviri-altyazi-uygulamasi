from __future__ import annotations

import sys

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from config import Settings
from overlay_ui import SubtitleOverlay
from pipeline import Pipeline
from tray_app import TrayApp

QUIT_JOIN_SECONDS = 2.0


class _PipelineBridge(QObject):

    result = Signal(object, str, str)
    status = Signal(object, str)
    failed = Signal(object)

    def __init__(self, is_current, on_result, on_status, on_failed) -> None:
        super().__init__()
        self._is_current = is_current
        self._on_result = on_result
        self._on_status = on_status
        self._on_failed = on_failed
        self.result.connect(self._deliver_result)
        self.status.connect(self._deliver_status)
        self.failed.connect(self._deliver_failed)

    def _deliver_result(self, pipeline, original: str, translated: str) -> None:
        if self._is_current(pipeline):
            self._on_result(original, translated)

    def _deliver_status(self, pipeline, message: str) -> None:
        if self._is_current(pipeline):
            self._on_status(message)

    def _deliver_failed(self, pipeline) -> None:
        if self._is_current(pipeline):
            self._on_failed(pipeline)


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    settings = Settings.load()

    overlay = SubtitleOverlay()
    overlay.set_locked(settings.locked)
    overlay.show()

    state: dict[str, Pipeline | None] = {"pipeline": None, "last": None}

    def on_pipeline_failed(pipeline: Pipeline) -> None:
        state["pipeline"] = None
        tray.mark_stopped()

    bridge = _PipelineBridge(
        is_current=lambda pipeline: state["pipeline"] is pipeline,
        on_result=overlay.set_subtitle,
        on_status=overlay.set_text,
        on_failed=on_pipeline_failed,
    )

    def start_pipeline() -> None:
        holder: list[Pipeline] = []
        pipeline = Pipeline(
            settings,
            on_result=lambda original, translated: bridge.result.emit(holder[0], original, translated),
            on_status=lambda message: bridge.status.emit(holder[0], message),
            on_fatal=bridge.failed.emit,
            previous=state["last"],
        )
        holder.append(pipeline)
        state["pipeline"] = pipeline
        state["last"] = pipeline
        pipeline.start()

    def stop_pipeline() -> None:
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.stop()
            state["pipeline"] = None
        overlay.set_text("Durduruldu")

    def shutdown() -> None:
        last = state["last"]
        if last is not None:
            last.stop()
            last.join(timeout=QUIT_JOIN_SECONDS)

    def toggle_lock(locked: bool) -> None:
        overlay.set_locked(locked)

    tray = TrayApp(
        settings=settings,
        on_start=start_pipeline,
        on_stop=stop_pipeline,
        on_toggle_lock=toggle_lock,
    )
    app.aboutToQuit.connect(shutdown)
    tray.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
