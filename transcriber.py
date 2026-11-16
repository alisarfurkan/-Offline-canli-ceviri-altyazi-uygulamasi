from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpu_env import register_nvidia_dll_dirs

register_nvidia_dll_dirs()

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str

DEFAULT_MODEL_SIZE = "large-v3-turbo"

COMPRESSION_RATIO_THRESHOLD = 2.4
CREDIT_PHRASES = (
    "altyazı m.k", "altyazi m.k", "amara.org", "subtitles by", "untertitel im auftrag", "sous-titres réalisés",
)


def _is_hallucination(segment) -> bool:
    if segment.compression_ratio > COMPRESSION_RATIO_THRESHOLD:
        return True
    text = segment.text.strip().lower()
    return any(phrase in text for phrase in CREDIT_PHRASES)


class Transcriber:
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if not _is_hallucination(segment)).strip()
        return TranscriptionResult(text=text, language=info.language)
