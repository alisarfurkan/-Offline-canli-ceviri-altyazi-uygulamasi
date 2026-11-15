from __future__ import annotations

import time

import numpy as np
import soundcard as sc

SAMPLE_RATE = 16000
DEFAULT_DEVICE_CHECK_SECONDS = 2.0


def list_output_devices() -> list[str]:
    return [speaker.name for speaker in sc.all_speakers()]


def _resolve_speaker(device_name: str | None):
    return sc.get_speaker(device_name) if device_name else sc.default_speaker()


def stream_frames(device_name: str | None = None, block_ms: int = 100, should_stop=None):
    speaker = _resolve_speaker(device_name)
    mic = sc.get_microphone(speaker.id, include_loopback=True)
    block_size = int(SAMPLE_RATE * block_ms / 1000)
    last_check = time.monotonic()
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while should_stop is None or not should_stop():
            frames = recorder.record(numframes=block_size)
            yield frames.reshape(-1).astype(np.float32)

            if device_name is None and time.monotonic() - last_check >= DEFAULT_DEVICE_CHECK_SECONDS:
                last_check = time.monotonic()
                if sc.default_speaker().id != speaker.id:
                    return
