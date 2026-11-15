from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import overlay_ui
import pipeline as pipeline_mod
import tray_app
import translator as translator_mod
from config import Settings
from overlay_ui import SubtitleOverlay
from transcriber import TranscriptionResult

APP = QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class Recorder:
    def __init__(self):
        self.statuses: list[str] = []
        self.results: list[tuple[str, str]] = []
        self.fatal: list[object] = []

    def make(self, settings=None, previous=None):
        return pipeline_mod.Pipeline(
            settings or Settings(),
            on_result=lambda o, t: self.results.append((o, t)),
            on_status=self.statuses.append,
            on_fatal=self.fatal.append,
            previous=previous,
        )


class BlockingTranscriber:

    release = threading.Event()
    created = 0

    def __init__(self, model_size=None):
        type(self).created += 1
        type(self).release.wait(5)


class PipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        BlockingTranscriber.release = threading.Event()
        BlockingTranscriber.created = 0
        self.rec = Recorder()

    def test_stop_during_load_does_not_start_listening(self):
        with mock.patch.object(pipeline_mod, "Transcriber", BlockingTranscriber), \
             mock.patch.object(pipeline_mod, "create_translator") as create_tr:
            p = self.rec.make()
            p.start()
            self.assertTrue(wait_until(lambda: BlockingTranscriber.created == 1))
            p.stop()
            BlockingTranscriber.release.set()
            p.join(timeout=3)

        self.assertEqual(self.rec.statuses, ["Modeller yükleniyor..."])
        self.assertEqual(p._workers, [])
        self.assertIsNone(p._transcriber)
        create_tr.assert_not_called()

    def test_load_failure_reports_fatal(self):
        with mock.patch.object(pipeline_mod, "Transcriber", side_effect=RuntimeError("CUDA yok")), \
             mock.patch("traceback.print_exc"):
            p = self.rec.make()
            p.start()
            p.join(timeout=3)

        self.assertTrue(any("Model yükleme hatası: CUDA yok" in s for s in self.rec.statuses))
        self.assertEqual(self.rec.fatal, [p])
        self.assertTrue(p.stopped)

    def test_new_pipeline_waits_for_previous_loader(self):
        noop = lambda self: None
        with mock.patch.object(pipeline_mod, "Transcriber", BlockingTranscriber), \
             mock.patch.object(pipeline_mod, "create_translator"), \
             mock.patch.multiple(pipeline_mod.Pipeline, _run_capture=noop, _run_vad=noop,
                                 _run_asr=noop, _run_mt=noop):
            old = self.rec.make()
            old.start()
            self.assertTrue(wait_until(lambda: BlockingTranscriber.created == 1))
            old.stop()

            new = self.rec.make(previous=old)
            new.start()
            time.sleep(0.2)
            self.assertEqual(BlockingTranscriber.created, 1)

            BlockingTranscriber.release.set()
            self.assertTrue(wait_until(lambda: BlockingTranscriber.created == 2))
            new.stop()
            new.join(timeout=3)

    def test_no_callbacks_after_stop(self):
        p = self.rec.make()
        p.stop()
        p._status("x")
        p._result("a", "b")
        self.assertEqual(self.rec.statuses, [])
        self.assertEqual(self.rec.results, [])


class StageTests(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()

    def _run_stage(self, p, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        return t

    def test_translation_failure_shows_original(self):
        p = self.rec.make(Settings(target_lang="tr"))
        p._translator = mock.Mock()
        p._translator.translate.side_effect = ValueError("NLLB için bilinmeyen dil kodu: 'uk'")
        t = self._run_stage(p, p._run_mt)
        with mock.patch("traceback.print_exc"):
            p._mt_q.put(("Привіт", "uk"))
            self.assertTrue(wait_until(lambda: self.rec.results))
        p.stop()
        t.join(2)
        self.assertEqual(self.rec.results, [("Привіт", "")])

    def test_same_language_is_not_translated(self):
        p = self.rec.make(Settings(target_lang="tr"))
        fake = p._translator = mock.Mock()
        t = self._run_stage(p, p._run_mt)
        p._mt_q.put(("Merhaba", "tr"))
        self.assertTrue(wait_until(lambda: self.rec.results))
        p.stop()
        t.join(2)
        self.assertEqual(self.rec.results, [("Merhaba", "")])
        fake.translate.assert_not_called()

    def test_stop_releases_models(self):
        p = self.rec.make()
        p._transcriber = mock.Mock()
        p._translator = mock.Mock()
        threads = [self._run_stage(p, p._run_asr), self._run_stage(p, p._run_mt)]
        p.stop()
        for t in threads:
            t.join(2)
        self.assertIsNone(p._transcriber)
        self.assertIsNone(p._translator)

    def test_stop_right_after_load_releases_models(self):
        with mock.patch.object(pipeline_mod, "Transcriber"), \
             mock.patch.object(pipeline_mod, "create_translator",
                               side_effect=lambda backend: (p.stop(), mock.Mock())[1]):
            p = self.rec.make()
            p.start()
            p.join(timeout=3)
        self.assertEqual(p._workers, [])
        self.assertIsNone(p._transcriber)
        self.assertIsNone(p._translator)

    def test_vad_init_failure_is_fatal(self):
        p = self.rec.make()
        with mock.patch.object(pipeline_mod, "VadSegmenter", side_effect=RuntimeError("onnx")), \
             mock.patch("traceback.print_exc"):
            p._run_vad()
        self.assertTrue(any("VAD yükleme hatası" in s for s in self.rec.statuses))
        self.assertEqual(self.rec.fatal, [p])

    def test_capture_retries_after_device_error(self):
        calls = {"n": 0}

        def fake_stream(device_name=None, should_stop=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("cihaz çıkarıldı")
            while not should_stop():
                yield np.zeros(1600, dtype=np.float32)
                time.sleep(0.01)

        p = self.rec.make()
        with mock.patch.object(pipeline_mod, "stream_frames", fake_stream), \
             mock.patch.object(pipeline_mod, "CAPTURE_RETRY_SECONDS", 0.05), \
             mock.patch("traceback.print_exc"):
            t = self._run_stage(p, p._run_capture)
            self.assertTrue(wait_until(lambda: not p._audio_q.empty()))
            p.stop()
            t.join(2)

        self.assertFalse(t.is_alive())
        self.assertEqual(calls["n"], 2)
        self.assertIn("cihaz çıkarıldı", self.rec.statuses[0])
        self.assertEqual(self.rec.statuses[1], "Dinleniyor...")


class ArgosTests(unittest.TestCase):
    def test_failed_pair_is_not_retried_every_sentence(self):
        tr = translator_mod.ArgosTranslator()
        with mock.patch.object(tr, "_has_path", return_value=False), \
             mock.patch.object(translator_mod.argostranslate.package, "update_package_index",
                               side_effect=OSError("internet yok")) as update:
            for _ in range(3):
                with self.assertRaises(Exception):
                    tr.translate("hello", "sv", "tr")
        self.assertEqual(update.call_count, 1)

    def test_failed_pair_is_retried_after_timeout(self):
        tr = translator_mod.ArgosTranslator()
        with mock.patch.object(tr, "_has_path", return_value=False), \
             mock.patch.object(translator_mod.argostranslate.package, "update_package_index",
                               side_effect=OSError("internet yok")) as update:
            with self.assertRaises(Exception):
                tr.translate("hello", "sv", "tr")
            with mock.patch.object(translator_mod, "ARGOS_RETRY_SECONDS", 0.0):
                with self.assertRaises(Exception):
                    tr.translate("hello", "sv", "tr")
        self.assertEqual(update.call_count, 2)


class BridgeTests(unittest.TestCase):
    def test_stale_pipeline_messages_are_dropped_in_gui_thread(self):
        import main as main_mod

        current = {"p": "yeni"}
        shown, failed = [], []
        bridge = main_mod._PipelineBridge(
            is_current=lambda p: current["p"] is p,
            on_result=lambda o, t: shown.append((o, t)),
            on_status=shown.append,
            on_failed=failed.append,
        )
        old, new = "eski", current["p"]

        t = threading.Thread(target=lambda: (
            bridge.result.emit(old, "a", "b"),
            bridge.status.emit(old, "x"),
            bridge.failed.emit(old),
            bridge.result.emit(new, "c", "d"),
            bridge.failed.emit(new),
        ))
        t.start()
        t.join()
        self.assertEqual(shown, [])
        APP.processEvents()
        self.assertEqual(shown, [("c", "d")])
        self.assertEqual(failed, [new])


class OverlayTests(unittest.TestCase):
    @mock.patch.object(overlay_ui, "WIDTH", 300)
    def test_dragged_position_survives_new_text(self):
        ov = SubtitleOverlay()
        ov.show()
        ov.move(150, 200)
        bottom = ov.y() + ov.height()
        ov.set_subtitle("x " * 200, "y " * 200)
        self.assertGreater(ov.height(), 120)
        self.assertEqual(ov.x(), 150)
        self.assertEqual(ov.y() + ov.height(), bottom)
        ov.set_text("kısa")
        self.assertEqual(ov.x(), 150)
        self.assertEqual(ov.y() + ov.height(), bottom)
        ov.close()

    def test_growing_bar_stays_on_screen(self):
        ov = SubtitleOverlay()
        ov.show()
        top = ov.screen().availableGeometry().top()
        ov.move(150, top)
        ov.set_subtitle("x " * 200, "y " * 200)
        self.assertGreaterEqual(ov.y(), top)
        ov.close()

    def test_lock_keeps_position(self):
        ov = SubtitleOverlay()
        ov.show()
        ov.move(150, 200)
        ov.set_locked(True)
        self.assertEqual((ov.x(), ov.y()), (150, 200))
        ov.close()

    def test_lock_makes_native_window_input_transparent(self):
        ov = SubtitleOverlay()
        ov.show()
        ov.set_locked(True)
        self.assertTrue(ov.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        self.assertTrue(ov.isVisible())
        ov.set_locked(False)
        self.assertFalse(ov.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        self.assertTrue(ov.isVisible())
        ov.close()

    def test_empty_translation_shows_original_as_main_line(self):
        ov = SubtitleOverlay()
        ov.set_subtitle("Hello", "")
        self.assertIn("Hello", ov.label.text())
        self.assertNotIn("#bbbbbb", ov.label.text())
        ov.close()


class TraySettingsTests(unittest.TestCase):
    def _make_tray(self):
        self.on_start = mock.Mock()
        self.on_stop = mock.Mock()
        settings = mock.Mock(spec=Settings)
        settings.locked = False
        tray = tray_app.TrayApp(settings, self.on_start, self.on_stop, mock.Mock())
        tray.start()
        self.on_start.reset_mock()
        return tray

    def test_cancel_keeps_listening(self):
        tray = self._make_tray()
        with mock.patch.object(tray_app, "SettingsDialog") as dlg:
            dlg.return_value.exec.return_value = 0
            tray._open_settings()
        self.on_stop.assert_not_called()
        self.on_start.assert_not_called()
        self.assertTrue(tray._running)
        tray.tray.hide()

    def test_ok_while_running_restarts(self):
        tray = self._make_tray()
        with mock.patch.object(tray_app, "SettingsDialog") as dlg, \
             mock.patch.object(tray_app, "QMessageBox") as box:
            dlg.return_value.exec.return_value = 1
            tray._open_settings()
        self.on_stop.assert_called_once()
        self.on_start.assert_called_once()
        self.assertTrue(tray._running)
        box.information.assert_not_called()
        tray.tray.hide()

    def test_mark_stopped_resets_menu(self):
        tray = self._make_tray()
        tray.mark_stopped()
        self.assertFalse(tray._running)
        self.assertEqual(tray.start_stop_action.text(), "Başlat")
        tray.tray.hide()


if __name__ == "__main__":
    unittest.main()
