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


class Transcriber:
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language: str | None = None, initial_prompt: str | None = None) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(text=text, language=info.language)
