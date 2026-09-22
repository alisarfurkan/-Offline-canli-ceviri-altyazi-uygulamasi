from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from config import Settings
from settings_dialog import SettingsDialog


def _make_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(30, 144, 255))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(4, 16, 56, 32, 8, 8)
    painter.setPen(QColor(255, 255, 255))
    font = painter.font()
    font.setBold(True)
    font.setPointSize(20)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "CC")
    painter.end()
    return QIcon(pixmap)


class TrayApp:
    def __init__(self, settings: Settings, on_start, on_stop, on_toggle_lock, on_settings_changed):
        self.settings = settings
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_toggle_lock = on_toggle_lock
        self.on_settings_changed = on_settings_changed
        self._running = False

        self.tray = QSystemTrayIcon(_make_icon())
        self.tray.setToolTip("Offline Canlı Altyazı")

        self.menu = QMenu()
        self.start_stop_action = self.menu.addAction("Başlat")
        self.start_stop_action.triggered.connect(self._toggle_start_stop)

        self.lock_action = self.menu.addAction("Konumu Kilitle")
        self.lock_action.setCheckable(True)
        self.lock_action.setChecked(settings.locked)
        self.lock_action.triggered.connect(self._toggle_lock)

        self.menu.addSeparator()
        settings_action = self.menu.addAction("Ayarlar...")
        settings_action.triggered.connect(self._open_settings)

        self.menu.addSeparator()
        quit_action = self.menu.addAction("Çıkış")
        quit_action.triggered.connect(QApplication.quit)

        self.tray.setContextMenu(self.menu)
        self.tray.show()

    def start(self) -> None:
        if not self._running:
            self._toggle_start_stop()

    def _toggle_start_stop(self) -> None:
        if self._running:
            self.on_stop()
            self.start_stop_action.setText("Başlat")
            self._running = False
        else:
            self.on_start()
            self.start_stop_action.setText("Durdur")
            self._running = True

    def _toggle_lock(self, checked: bool) -> None:
        self.settings.locked = checked
        self.settings.save()
        self.on_toggle_lock(checked)

    def _open_settings(self) -> None:
        was_running = self._running
        if was_running:
            self.on_stop()
            self.start_stop_action.setText("Başlat")
            self._running = False

        dialog = SettingsDialog(self.settings)
        if dialog.exec():
            dialog.apply_to(self.settings)
            self.settings.save()
            self.on_settings_changed()
            QMessageBox.information(None, "Ayarlar", "Ayarlar kaydedildi. Başlat'a basarak yeni ayarlarla dinlemeye başlayabilirsiniz.")
