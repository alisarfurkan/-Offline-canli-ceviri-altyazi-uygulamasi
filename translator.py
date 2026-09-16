from __future__ import annotations

import os

os.environ.setdefault("ARGOS_STANZA_AVAILABLE", "0")

from abc import ABC, abstractmethod
from pathlib import Path

import argostranslate.package
import argostranslate.translate

NLLB_MODEL_DIR = Path(__file__).parent / "models" / "nllb-ct2"

ISO_TO_FLORES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
}


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        ...


class ArgosTranslator(Translator):
    def __init__(self) -> None:
        self._ensured_pairs: set[tuple[str, str]] = set()

    def _has_path(self, source_lang: str, target_lang: str) -> bool:
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed if lang.code == source_lang), None)
        to_lang = next((lang for lang in installed if lang.code == target_lang), None)
        if from_lang is None or to_lang is None:
            return False
        return from_lang.get_translation(to_lang) is not None

    def ensure_package(self, source_lang: str, target_lang: str) -> None:
        pair = (source_lang, target_lang)
        if pair in self._ensured_pairs:
            return
        if self._has_path(source_lang, target_lang):
            self._ensured_pairs.add(pair)
            return

        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        installed_pairs = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}

        needed_codes = {source_lang, target_lang, "en"}
        for pkg in available:
            if pkg.from_code in needed_codes and pkg.to_code in needed_codes and pkg.from_code != pkg.to_code:
                if (pkg.from_code, pkg.to_code) not in installed_pairs:
                    path = pkg.download()
                    argostranslate.package.install_from_path(path)

        if not self._has_path(source_lang, target_lang):
            raise RuntimeError(
                f"Argos Translate: {source_lang} -> {target_lang} için (doğrudan veya İngilizce "
                "üzerinden) bir çeviri yolu bulunamadı."
            )
        self._ensured_pairs.add(pair)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return ""
        self.ensure_package(source_lang, target_lang)
        return argostranslate.translate.translate(text, source_lang, target_lang)


class NllbCt2Translator(Translator):
    def __init__(self, model_dir: Path = NLLB_MODEL_DIR, device: str = "cuda", compute_type: str = "float16") -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"NLLB CTranslate2 modeli bulunamadı: {model_dir}\n"
                "Önce şunu çalıştırın:\n"
                "  ct2-transformers-converter --model facebook/nllb-200-distilled-600M "
                f"--output_dir {model_dir} --quantization float16"
            )
        from gpu_env import register_nvidia_dll_dirs

        register_nvidia_dll_dirs()

        import ctranslate2
        from transformers import AutoTokenizer

        self._translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        tokenizer_dir = model_dir / "tokenizer"
        tokenizer_source = tokenizer_dir if tokenizer_dir.is_dir() else "facebook/nllb-200-distilled-600M"
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    @staticmethod
    def _flores_code(iso_code: str) -> str:
        flores = ISO_TO_FLORES.get(iso_code)
        if flores is None:
            raise ValueError(f"NLLB için bilinmeyen dil kodu: {iso_code!r}")
        return flores

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return ""

        self._tokenizer.src_lang = self._flores_code(source_lang)
        target_flores = self._flores_code(target_lang)

        source_tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[target_flores]],
            beam_size=4,
        )
        target_tokens = results[0].hypotheses[0][1:]
        target_ids = self._tokenizer.convert_tokens_to_ids(target_tokens)
        return self._tokenizer.decode(target_ids, skip_special_tokens=True).strip()


def create_translator(backend: str) -> Translator:
    if backend == "nllb":
        return NllbCt2Translator()
    if backend == "argos":
        return ArgosTranslator()
    raise ValueError(f"Bilinmeyen çeviri backend'i: {backend!r}")
