from __future__ import annotations

import html
import sys

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QTextDocument
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

WIDTH = 1000
SCREEN_SIDE_MARGIN = 20
MIN_HEIGHT = 120
MAX_HEIGHT = 320
MARGIN_H, MARGIN_V = 24, 16
BOTTOM_MARGIN = 60
SUBTITLE_TIMEOUT_MS = 13000

if sys.platform == "win32":
    import ctypes

    _HWND_TOPMOST = -1
    _SWP_NOMOVE = 0x0002
    _SWP_NOSIZE = 0x0001
    _SWP_NOACTIVATE = 0x0010


class SubtitleOverlay(QWidget):
    text_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._locked = False
        self._drag_offset = QPoint()

        self.label = QLabel("Altyazı bekleniyor...", self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: white; background: transparent;")
        self.label.setTextFormat(Qt.TextFormat.RichText)
        font = QFont("Segoe UI", 20, QFont.Weight.DemiBold)
        self.label.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN_H, MARGIN_V, MARGIN_H, MARGIN_V)
        layout.addWidget(self.label)

        self.text_changed.connect(self._apply_text)

        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(lambda: self.text_changed.emit(""))

        self.resize(self._bar_width(), MIN_HEIGHT)
        self._reposition_bottom_anchored()

        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._reassert_topmost)
        self._topmost_timer.start(500)

    def _reassert_topmost(self) -> None:
        if sys.platform != "win32":
            return
        hwnd = int(self.winId())
        ctypes.windll.user32.SetWindowPos(
            hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE
        )

    def _available_geometry(self):
        return (self.screen() or QApplication.primaryScreen()).availableGeometry()

    def _bar_width(self) -> int:
        return max(200, min(WIDTH, self._available_geometry().width() - 2 * SCREEN_SIDE_MARGIN))

    def _reposition_bottom_anchored(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        bottom_y = screen.y() + screen.height() - BOTTOM_MARGIN
        x = screen.x() + (screen.width() - self.width()) // 2
        y = bottom_y - self.height()
        self.move(x, y)

    def _apply_text(self, rich_text: str) -> None:
        self.label.setText(rich_text)

        doc = QTextDocument()
        doc.setDefaultFont(self.label.font())
        doc.setHtml(rich_text)
        width = self._bar_width()
        doc.setTextWidth(width - 2 * MARGIN_H)
        needed_height = int(doc.size().height()) + 2 * MARGIN_V
        new_height = max(MIN_HEIGHT, min(MAX_HEIGHT, needed_height))

        if new_height != self.height() or width != self.width():
            bottom = self.y() + self.height()
            self.resize(width, new_height)
            screen = self._available_geometry()
            y = max(screen.top(), min(bottom - new_height, screen.y() + screen.height() - new_height))
            x = max(screen.left(), min(self.x(), screen.x() + screen.width() - width))
            self.move(x, y)
        self.update()

    def set_text(self, text: str) -> None:
        self._clear_timer.stop()
        self.text_changed.emit(html.escape(text))

    def set_subtitle(self, original: str, translated: str) -> None:
        if not translated:
            rich = f"<div style='font-size:22px;font-weight:600;color:white;'>{html.escape(original)}</div>"
        else:
            rich = (
                f"<div style='font-size:13px;color:#bbbbbb;'>{html.escape(original)}</div>"
                f"<div style='font-size:22px;font-weight:600;color:white;'>{html.escape(translated)}</div>"
            )
        self.text_changed.emit(rich)
        self._clear_timer.start(SUBTITLE_TIMEOUT_MS)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, locked)
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, locked)
        if was_visible:
            self.show()

    def paintEvent(self, event) -> None:
        empty = not self.label.text()
        if empty and self._locked:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 16, 16)
        painter.fillPath(path, QColor(0, 0, 0, 40 if empty else 160))

    def mousePressEvent(self, event) -> None:
        if not self._locked and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if not self._locked and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
