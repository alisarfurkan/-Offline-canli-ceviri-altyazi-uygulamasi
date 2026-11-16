from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
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
        if not CONFIG_PATH.exists():
            return cls()
        try:
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            print(f"config.yaml okunamadı, varsayılan ayarlar kullanılıyor: {exc}", file=sys.stderr)
            return cls()
        if not isinstance(data, dict):
            print("config.yaml beklenen biçimde değil, varsayılan ayarlar kullanılıyor", file=sys.stderr)
            return cls()
        defaults = cls()
        values = {}
        for field in fields(cls):
            if field.name not in data:
                continue
            value, default = data[field.name], getattr(defaults, field.name)
            allowed = (str, type(None)) if default is None else (type(default),)
            if type(value) in allowed:
                values[field.name] = value
            else:
                print(f"config.yaml: '{field.name}' değeri geçersiz ({value!r}), varsayılan kullanılıyor", file=sys.stderr)
        return cls(**values)

    def save(self) -> None:
        CONFIG_PATH.write_text(yaml.safe_dump(asdict(self), allow_unicode=True), encoding="utf-8")
