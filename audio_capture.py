from __future__ import annotations

import numpy as np
import soundcard as sc

SAMPLE_RATE = 16000


def list_output_devices() -> list[str]:
    return [speaker.name for speaker in sc.all_speakers()]


def get_loopback_microphone(device_name: str | None = None) -> sc.Microphone:
    speaker = sc.get_speaker(device_name) if device_name else sc.default_speaker()
    return sc.get_microphone(speaker.name, include_loopback=True)


def record_seconds(seconds: float, device_name: str | None = None) -> np.ndarray:
    mic = get_loopback_microphone(device_name)
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        frames = recorder.record(numframes=int(seconds * SAMPLE_RATE))
    return frames.reshape(-1).astype(np.float32)


def stream_frames(device_name: str | None = None, block_ms: int = 100):
    mic = get_loopback_microphone(device_name)
    block_size = int(SAMPLE_RATE * block_ms / 1000)
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while True:
            frames = recorder.record(numframes=block_size)
            yield frames.reshape(-1).astype(np.float32)
