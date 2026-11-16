from __future__ import annotations

from collections import deque

import numpy as np
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512
PRE_ROLL_FRAMES = 3
CUT_SEARCH_SECONDS = 1.0


class VadSegmenter:
    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 600,
        max_utterance_seconds: float = 10.0,
    ):
        self.model = load_silero_vad(onnx=True)
        self.iterator = VADIterator(
            self.model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_buffer: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
        self._in_speech = False
        self._max_utterance_samples = int(max_utterance_seconds * SAMPLE_RATE)

    def _finish_utterance(self) -> np.ndarray:
        utterance = np.concatenate(self._speech_buffer) if self._speech_buffer else np.zeros(0, dtype=np.float32)
        self._speech_buffer = []
        self._in_speech = False
        return utterance

    def _cut_long_utterance(self) -> np.ndarray:
        search = min(len(self._speech_buffer), max(1, int(CUT_SEARCH_SECONDS * SAMPLE_RATE / VAD_FRAME_SAMPLES)))
        tail = self._speech_buffer[-search:]
        quietest = min(range(search), key=lambda i: float(np.mean(tail[i] ** 2)))
        cut = len(self._speech_buffer) - search + quietest + 1
        utterance = np.concatenate(self._speech_buffer[:cut])
        self._speech_buffer = self._speech_buffer[cut:]
        return utterance

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        finished: list[np.ndarray] = []
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32)])

        while self._pending.size >= VAD_FRAME_SAMPLES:
            frame = self._pending[:VAD_FRAME_SAMPLES]
            self._pending = self._pending[VAD_FRAME_SAMPLES:]

            consumed = self._in_speech
            if consumed:
                self._speech_buffer.append(frame)

            event = self.iterator(frame, return_seconds=False)
            if event is not None:
                if "start" in event and not self._in_speech:
                    self._in_speech = True
                    self._speech_buffer = [*self._pre_roll, frame]
                    self._pre_roll.clear()
                    consumed = True
                elif "end" in event and self._in_speech:
                    finished.append(self._finish_utterance())

            if not consumed:
                self._pre_roll.append(frame)

            if self._in_speech and sum(f.size for f in self._speech_buffer) >= self._max_utterance_samples:
                finished.append(self._cut_long_utterance())

        return finished

    def reset(self) -> None:
        self.iterator.reset_states()
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_buffer = []
        self._pre_roll.clear()
        self._in_speech = False
