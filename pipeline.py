from __future__ import annotations

import faulthandler
import queue
import sys
import threading
import time
import traceback
from typing import Callable

from audio_capture import SAMPLE_RATE, stream_frames
from config import Settings
from transcriber import Transcriber
from translator import Translator, create_translator
from vad_segmenter import VadSegmenter

OnResult = Callable[[str, str], None]
OnStatus = Callable[[str], None]
OnFatal = Callable[["Pipeline"], None]

CAPTURE_RETRY_SECONDS = 2.0
WORKER_JOIN_SECONDS = 5.0
MAX_UTTERANCE_SECONDS = 10.0
SLOW_ASR_SECONDS = 30.0


def _arm_slow_asr_dump() -> bool:
    try:
        faulthandler.dump_traceback_later(SLOW_ASR_SECONDS, file=sys.stderr)
        return True
    except (AttributeError, ValueError, RuntimeError, OSError):
        return False


class Pipeline:

    def __init__(
        self,
        settings: Settings,
        on_result: OnResult,
        on_status: OnStatus | None = None,
        on_fatal: OnFatal | None = None,
        previous: "Pipeline | None" = None,
    ):
        self.settings = settings
        self.on_result = on_result
        self.on_status = on_status or (lambda msg: None)
        self.on_fatal = on_fatal or (lambda pipeline: None)
        self._previous = previous

        self._stop_event = threading.Event()
        self._loader: threading.Thread | None = None
        self._workers: list[threading.Thread] = []

        self._audio_q: "queue.Queue" = queue.Queue(maxsize=100)
        self._asr_q: "queue.Queue" = queue.Queue(maxsize=5)
        self._mt_q: "queue.Queue" = queue.Queue(maxsize=20)

        self._transcriber: Transcriber | None = None
        self._translator: Translator | None = None

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def start(self) -> None:
        self._status("Modeller yükleniyor...")

        self._loader = threading.Thread(target=self._load_and_run, daemon=True)
        self._loader.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        if self._loader is not None:
            self._loader.join(timeout)
        if deadline is None:
            deadline = time.monotonic() + WORKER_JOIN_SECONDS
        for t in list(self._workers):
            t.join(max(0.0, deadline - time.monotonic()))

    def _status(self, message: str) -> None:
        if not self.stopped:
            self.on_status(message)

    def _result(self, original: str, translated: str) -> None:
        if not self.stopped:
            self.on_result(original, translated)

    def _fail(self, message: str) -> None:
        traceback.print_exc()
        if self.stopped:
            return
        self.on_status(message)
        self._stop_event.set()
        self.on_fatal(self)

    def _release_models(self) -> None:
        self._transcriber = None
        self._translator = None

    def _load_and_run(self) -> None:
        if self._previous is not None:
            self._previous.join()
            self._previous = None

        try:
            if not self.stopped:
                self._transcriber = Transcriber(model_size=self.settings.model_size)
            if not self.stopped:
                self._translator = create_translator(self.settings.translator_backend)
        except Exception as exc:
            self._fail(f"Model yükleme hatası: {exc}")
            self._release_models()
            return

        if self.stopped:
            self._release_models()
            return

        self._status("Dinleniyor...")

        for target in (self._run_capture, self._run_vad, self._run_asr, self._run_mt):
            t = threading.Thread(target=target, daemon=True)
            self._workers.append(t)
            t.start()

    def _run_capture(self) -> None:
        error_shown = False
        while not self.stopped:
            try:
                for block in stream_frames(device_name=self.settings.device_name, should_stop=lambda: self.stopped):
                    if error_shown:
                        self._status("Dinleniyor...")
                        error_shown = False
                    try:
                        self._audio_q.put_nowait(block)
                    except queue.Full:
                        try:
                            self._audio_q.get_nowait()
                        except queue.Empty:
                            pass
                        self._audio_q.put_nowait(block)
            except Exception as exc:
                traceback.print_exc()
                if not error_shown:
                    self._status(f"Ses yakalama hatası: {exc} (yeniden deneniyor...)")
                    error_shown = True
                self._stop_event.wait(CAPTURE_RETRY_SECONDS)

    def _run_vad(self) -> None:
        try:
            segmenter = VadSegmenter(min_silence_duration_ms=600, max_utterance_seconds=MAX_UTTERANCE_SECONDS)
        except Exception as exc:
            self._fail(f"VAD yükleme hatası: {exc}")
            return

        while not self.stopped:
            try:
                block = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                utterances = segmenter.push(block)
            except Exception as exc:
                traceback.print_exc()
                self._status(f"VAD hatası: {exc}")
                segmenter.reset()
                continue
            for utterance in utterances:
                if utterance.size / SAMPLE_RATE < 0.2:
                    continue
                try:
                    self._asr_q.put_nowait(utterance)
                except queue.Full:
                    pass

    def _run_asr(self) -> None:
        try:
            self._asr_loop()
        finally:
            self._transcriber = None

    def _asr_loop(self) -> None:
        while not self.stopped:
            try:
                utterance = self._asr_q.get(timeout=0.5)
            except queue.Empty:
                continue
            language = None if self.settings.source_lang == "auto" else self.settings.source_lang
            armed = _arm_slow_asr_dump()
            try:
                result = self._transcriber.transcribe(utterance, language=language)
            except Exception as exc:
                traceback.print_exc()
                self._status(f"Tanıma hatası: {exc}")
                continue
            finally:
                if armed:
                    faulthandler.cancel_dump_traceback_later()
            if result.text:
                try:
                    self._mt_q.put_nowait((result.text, result.language))
                except queue.Full:
                    pass

    def _run_mt(self) -> None:
        try:
            self._mt_loop()
        finally:
            self._translator = None

    def _mt_loop(self) -> None:
        logged_pairs: set[tuple[str, str]] = set()
        while not self.stopped:
            try:
                text, detected_source = self._mt_q.get(timeout=0.5)
            except queue.Empty:
                continue
            target = self.settings.target_lang
            if detected_source == target:
                self._result(text, "")
                continue
            try:
                translated = self._translator.translate(text, detected_source, target)
            except Exception:
                pair = (detected_source, target)
                if pair not in logged_pairs:
                    logged_pairs.add(pair)
                    traceback.print_exc()
                translated = ""
            self._result(text, translated)
