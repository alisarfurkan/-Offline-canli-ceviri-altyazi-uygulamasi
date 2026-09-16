from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from audio_capture import stream_frames
from config import Settings
from transcriber import Transcriber
from translator import Translator, create_translator
from vad_segmenter import VadSegmenter

OnResult = Callable[[str, str], None]
OnStatus = Callable[[str], None]


@dataclass
class _Utterance:
    audio: object


class Pipeline:
    def __init__(self, settings: Settings, on_result: OnResult, on_status: OnStatus | None = None):
        self.settings = settings
        self.on_result = on_result
        self.on_status = on_status or (lambda msg: None)

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._audio_q: "queue.Queue" = queue.Queue(maxsize=100)
        self._asr_q: "queue.Queue" = queue.Queue(maxsize=5)
        self._mt_q: "queue.Queue" = queue.Queue(maxsize=20)

        self._transcriber: Transcriber | None = None
        self._translator: Translator | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self.on_status("Modeller yükleniyor...")

        loader = threading.Thread(target=self._load_and_run, daemon=True)
        loader.start()
        self._threads.append(loader)

    def _load_and_run(self) -> None:
        try:
            self._transcriber = Transcriber(model_size=self.settings.model_size)
            self._translator = create_translator(self.settings.translator_backend)
        except Exception as exc:
            self.on_status(f"Model yükleme hatası: {exc}")
            return

        self.on_status("Dinleniyor...")

        capture_thread = threading.Thread(target=self._run_capture, daemon=True)
        vad_thread = threading.Thread(target=self._run_vad, daemon=True)
        asr_thread = threading.Thread(target=self._run_asr, daemon=True)
        mt_thread = threading.Thread(target=self._run_mt, daemon=True)

        for t in (capture_thread, vad_thread, asr_thread, mt_thread):
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        self.on_status("Durduruldu")

    def _run_capture(self) -> None:
        try:
            for block in stream_frames(device_name=self.settings.device_name):
                if self._stop_event.is_set():
                    return
                try:
                    self._audio_q.put_nowait(block)
                except queue.Full:
                    try:
                        self._audio_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._audio_q.put_nowait(block)
        except Exception as exc:
            self.on_status(f"Ses yakalama hatası: {exc}")

    def _run_vad(self) -> None:
        segmenter = VadSegmenter(min_silence_duration_ms=600, max_utterance_seconds=10.0)
        while not self._stop_event.is_set():
            try:
                block = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            for utterance in segmenter.push(block):
                if utterance.size / 16000 < 0.2:
                    continue
                try:
                    self._asr_q.put_nowait(utterance)
                except queue.Full:
                    pass

    def _run_asr(self) -> None:
        while not self._stop_event.is_set():
            try:
                utterance = self._asr_q.get(timeout=0.5)
            except queue.Empty:
                continue
            language = None if self.settings.source_lang == "auto" else self.settings.source_lang
            result = self._transcriber.transcribe(utterance, language=language)
            if result.text:
                try:
                    self._mt_q.put_nowait((result.text, result.language))
                except queue.Full:
                    pass

    def _run_mt(self) -> None:
        while not self._stop_event.is_set():
            try:
                text, detected_source = self._mt_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                translated = self._translator.translate(text, detected_source, self.settings.target_lang)
            except Exception as exc:
                self.on_status(f"Çeviri hatası: {exc}")
                continue
            self.on_result(text, translated)
