from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class Settings:
    source_lang: str = "en"
    target_lang: str = "tr"
    device_name: str | None = None
    model_size: str = "large-v3-turbo"
    translator_backend: str = "nllb"
    locked: bool = False

    @classmethod
    def load(cls) -> "Settings":
        if CONFIG_PATH.exists():
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            return cls(**{**asdict(cls()), **data})
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(yaml.safe_dump(asdict(self), allow_unicode=True), encoding="utf-8")
