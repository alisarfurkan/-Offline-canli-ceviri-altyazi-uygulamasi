from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout

from audio_capture import list_output_devices
from config import Settings

LANGUAGE_CHOICES = [
    ("auto", "Otomatik algıla"),
    ("en", "İngilizce"),
    ("de", "Almanca"),
    ("fr", "Fransızca"),
    ("es", "İspanyolca"),
    ("it", "İtalyanca"),
    ("pt", "Portekizce"),
    ("ru", "Rusça"),
    ("ja", "Japonca"),
    ("ko", "Korece"),
    ("zh", "Çince"),
    ("ar", "Arapça"),
    ("nl", "Felemenkçe"),
    ("pl", "Lehçe"),
    ("tr", "Türkçe"),
]

MODEL_CHOICES = ["large-v3-turbo", "distil-large-v3", "medium", "small", "base", "tiny"]


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.settings = settings

        self.source_combo = QComboBox()
        for code, label in LANGUAGE_CHOICES:
            self.source_combo.addItem(f"{label} ({code})", code)
        self._select_by_data(self.source_combo, settings.source_lang)

        self.target_combo = QComboBox()
        for code, label in LANGUAGE_CHOICES:
            if code == "auto":
                continue
            self.target_combo.addItem(f"{label} ({code})", code)
        self._select_by_data(self.target_combo, settings.target_lang)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Varsayılan çıkış cihazı", None)
        for name in list_output_devices():
            self.device_combo.addItem(name, name)
        self._select_by_data(self.device_combo, settings.device_name)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        if settings.model_size in MODEL_CHOICES:
            self.model_combo.setCurrentText(settings.model_size)

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("NLLB-200 (GPU, daha kaliteli)", "nllb")
        self.backend_combo.addItem("Argos Translate (CPU, basit)", "argos")
        self._select_by_data(self.backend_combo, settings.translator_backend)

        form = QFormLayout(self)
        form.addRow("Kaynak dil (konuşulan dil):", self.source_combo)
        form.addRow("Hedef dil (altyazı dili):", self.target_combo)
        form.addRow("Dinlenecek ses cihazı:", self.device_combo)
        form.addRow("Whisper model boyutu:", self.model_combo)
        form.addRow("Çeviri motoru:", self.backend_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _select_by_data(combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def apply_to(self, settings: Settings) -> None:
        settings.source_lang = self.source_combo.currentData()
        settings.target_lang = self.target_combo.currentData()
        settings.device_name = self.device_combo.currentData()
        settings.model_size = self.model_combo.currentText()
        settings.translator_backend = self.backend_combo.currentData()
