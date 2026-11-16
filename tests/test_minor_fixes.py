from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtWidgets import QApplication

import audio_capture
import config
import gpu_env
import main as main_mod
import overlay_ui
import settings_dialog
import transcriber as transcriber_mod
import vad_segmenter
from config import Settings

APP = QApplication.instance() or QApplication([])
FRAME = vad_segmenter.VAD_FRAME_SAMPLES


def process_events_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class ConfigLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.yaml"
        patcher = mock.patch.object(config, "CONFIG_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_unknown_key_is_ignored(self):
        self.path.write_text("target_lang: de\neski_ayar: 1\n", encoding="utf-8")
        settings = Settings.load()
        self.assertEqual(settings.target_lang, "de")

    def test_corrupt_file_falls_back_to_defaults(self):
        self.path.write_text("target_lang: [kapanmamış\n", encoding="utf-8")
        self.assertEqual(Settings.load(), Settings())

    def test_non_mapping_file_falls_back_to_defaults(self):
        self.path.write_text("- bir\n- liste\n", encoding="utf-8")
        self.assertEqual(Settings.load(), Settings())

    def test_wrong_value_types_fall_back_per_field(self):
        self.path.write_text("locked: evet\nmodel_size: 3\nsource_lang:\ntarget_lang: de\n", encoding="utf-8")
        settings = Settings.load()
        defaults = Settings()
        self.assertIs(settings.locked, defaults.locked)
        self.assertEqual(settings.model_size, defaults.model_size)
        self.assertEqual(settings.source_lang, defaults.source_lang)
        self.assertEqual(settings.target_lang, "de")

    def test_device_name_may_be_none_or_text(self):
        self.path.write_text("device_name: Hoparlör\n", encoding="utf-8")
        self.assertEqual(Settings.load().device_name, "Hoparlör")
        self.path.write_text("device_name: null\n", encoding="utf-8")
        self.assertIsNone(Settings.load().device_name)
        self.path.write_text("device_name: 5\n", encoding="utf-8")
        self.assertIsNone(Settings.load().device_name)


class GpuEnvTests(unittest.TestCase):
    def test_dll_dirs_found_outside_venv_layout(self):
        with mock.patch.object(gpu_env, "_done", False), \
                mock.patch.object(gpu_env.sys, "executable", r"C:\yok\python.exe"), \
                mock.patch.object(gpu_env.os, "add_dll_directory", create=True) as add_dir, \
                mock.patch.dict(gpu_env.os.environ, {"PATH": ""}):
            gpu_env.register_nvidia_dll_dirs()
        added = [Path(call.args[0]) for call in add_dir.call_args_list]
        self.assertTrue(added, "hiç DLL klasörü eklenmedi")
        self.assertTrue(all(p.name == "bin" and p.parent.parent.name == "nvidia" for p in added))


class ScriptedIterator:

    def __init__(self, events: dict[int, dict]):
        self.events = events
        self.index = 0
        self.triggered = False

    def __call__(self, frame, return_seconds=False):
        event = self.events.get(self.index)
        self.index += 1
        return event

    def reset_states(self):
        pass


def make_segmenter(events, max_seconds=10.0):
    seg = vad_segmenter.VadSegmenter(max_utterance_seconds=max_seconds)
    seg.iterator = ScriptedIterator(events)
    return seg


def frames(values):
    return np.concatenate([np.full(FRAME, v, dtype=np.float32) for v in values])


class VadTests(unittest.TestCase):
    def test_audio_just_before_start_is_kept(self):
        seg = make_segmenter({5: {"start": 0}, 8: {"end": 0}})
        out = seg.push(frames([i / 100 for i in range(10)]))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0][0]), 0.02, places=5)

    def test_long_utterance_is_cut_at_quietest_frame(self):
        max_frames = 20
        seg = make_segmenter({0: {"start": 0}}, max_seconds=max_frames * FRAME / vad_segmenter.SAMPLE_RATE)
        values = [1 + i / 100 for i in range(max_frames)]
        values[15] = 0.0
        out = seg.push(frames(values))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].size, 16 * FRAME)
        seg.iterator.events[seg.iterator.index] = {"end": 0}
        rest = seg.push(frames([2.0]))
        self.assertEqual(len(rest), 1)
        self.assertAlmostEqual(float(rest[0][0]), 1.16, places=5)


def segment(text, no_speech=0.05, logprob=-0.3, ratio=1.5):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob, compression_ratio=ratio)


class HallucinationFilterTests(unittest.TestCase):
    def transcribe(self, segments):
        t = transcriber_mod.Transcriber.__new__(transcriber_mod.Transcriber)
        t.model = mock.Mock()
        t.model.transcribe.return_value = (iter(segments), SimpleNamespace(language="en"))
        return t.transcribe(np.zeros(16000, dtype=np.float32)).text

    def test_normal_speech_is_kept(self):
        self.assertEqual(self.transcribe([segment(" Hello there.")]), "Hello there.")

    def test_repetitive_output_is_dropped(self):
        self.assertEqual(self.transcribe([segment(" la la la", ratio=3.1), segment(" la la la", ratio=3.1)]), "")

    def test_subtitle_credit_lines_are_dropped(self):
        for credit in (" Altyazı M.K.", " ALTYAZI M.K.", " Subtitles by the Amara.org community"):
            with self.subTest(credit=credit):
                self.assertEqual(self.transcribe([segment(credit)]), "")


class OverlayTimeoutTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(overlay_ui, "SUBTITLE_TIMEOUT_MS", 50, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ov = overlay_ui.SubtitleOverlay()
        self.ov.show()
        self.addCleanup(self.ov.close)

    def test_subtitle_clears_after_silence(self):
        self.ov.set_subtitle("hello", "merhaba")
        self.assertTrue(process_events_until(lambda: self.ov.label.text() == ""))

    def alpha_at_center(self):
        image = self.ov.grab().toImage()
        return image.pixelColor(image.width() // 2, image.height() // 2).alpha()

    def test_cleared_bar_is_transparent_when_locked(self):
        self.ov.set_locked(True)
        self.ov.set_subtitle("hello", "merhaba")
        self.assertTrue(process_events_until(lambda: self.ov.label.text() == ""))
        self.assertEqual(self.alpha_at_center(), 0)

    def test_cleared_bar_stays_grabbable_when_unlocked(self):
        self.ov.set_subtitle("hello", "merhaba")
        self.assertTrue(process_events_until(lambda: self.ov.label.text() == ""))
        self.assertGreater(self.alpha_at_center(), 0)

    def test_status_text_is_not_cleared(self):
        self.ov.set_subtitle("hello", "merhaba")
        self.ov.set_text("Durduruldu")
        process_events_until(lambda: False, timeout=0.3)
        self.assertEqual(self.ov.label.text(), "Durduruldu")


class OverlayTimeoutValueTests(unittest.TestCase):
    def test_timeout_is_longer_than_longest_utterance(self):
        import pipeline as pipeline_mod

        self.assertGreater(overlay_ui.SUBTITLE_TIMEOUT_MS / 1000, pipeline_mod.MAX_UTTERANCE_SECONDS + 2)


class OverlayWidthTests(unittest.TestCase):
    def test_bar_fits_narrow_screen(self):
        ov = overlay_ui.SubtitleOverlay()
        ov.show()
        screen = ov.screen().availableGeometry()
        ov.set_subtitle("x " * 200, "y " * 200)
        self.assertLessEqual(ov.width(), screen.width())
        self.assertGreaterEqual(ov.x(), screen.left())
        self.assertLessEqual(ov.x() + ov.width(), screen.right() + 1)
        ov.close()


class SettingsDialogTests(unittest.TestCase):
    def test_custom_model_size_survives_ok(self):
        settings = Settings(model_size="large-v2")
        with mock.patch.object(settings_dialog, "list_output_devices", return_value=[]):
            dialog = settings_dialog.SettingsDialog(settings)
        dialog.apply_to(settings)
        self.assertEqual(settings.model_size, "large-v2")


@unittest.skipUnless(sys.platform == "win32", "Windows named mutex")
class SingleInstanceTests(unittest.TestCase):
    def setUp(self):
        self.name = f"altyazi-test-{os.getpid()}-{time.monotonic_ns()}"

    def test_second_instance_is_refused(self):
        first = main_mod.acquire_single_instance(self.name)
        self.assertIsNotNone(first)
        self.assertIsNone(main_mod.acquire_single_instance(self.name))
        main_mod.release_single_instance(first)
        third = main_mod.acquire_single_instance(self.name)
        self.assertIsNotNone(third)
        main_mod.release_single_instance(third)

    def test_crashed_instance_does_not_block_next_launch(self):
        import subprocess

        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import os, sys, win32event; h = win32event.CreateMutex(None, False, sys.argv[1]);"
             " print('ready', flush=True); sys.stdin.readline(); os._exit(1)", self.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            self.assertIsNone(main_mod.acquire_single_instance(self.name))
            holder.stdin.write("die\n")
            holder.stdin.flush()
            holder.wait(10)
            handle = main_mod.acquire_single_instance(self.name)
            self.assertIsNotNone(handle)
            main_mod.release_single_instance(handle)
        finally:
            if holder.poll() is None:
                holder.kill()
            holder.stdin.close()
            holder.stdout.close()


class FakeRecorder:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        return np.zeros((numframes, 1), dtype=np.float32)


class StreamFramesTests(unittest.TestCase):
    def test_returns_when_default_device_changes(self):
        fake_sc = mock.Mock()
        fake_sc.default_speaker.side_effect = [SimpleNamespace(id=i) for i in ("A", "A", "B")]
        fake_sc.get_microphone.return_value.recorder.return_value = FakeRecorder()
        count = []
        with mock.patch.object(audio_capture, "sc", fake_sc), \
                mock.patch.object(audio_capture, "DEFAULT_DEVICE_CHECK_SECONDS", 0):
            for _ in audio_capture.stream_frames(should_stop=lambda: len(count) >= 50):
                count.append(1)
        fake_sc.get_microphone.assert_called_once_with("A", include_loopback=True)
        self.assertEqual(len(count), 2)


class SlowAsrWatchdogTests(unittest.TestCase):
    def run_asr_once(self, transcribe_seconds):
        import pipeline as pipeline_mod

        results = []
        p = pipeline_mod.Pipeline(Settings(), on_result=lambda o, t: None)
        p._transcriber = mock.Mock()

        def slow(audio, language=None):
            time.sleep(transcribe_seconds)
            results.append(1)
            return transcriber_mod.TranscriptionResult(text="", language="en")

        p._transcriber.transcribe.side_effect = slow
        with tempfile.TemporaryFile("w+") as log, \
                mock.patch.object(pipeline_mod, "SLOW_ASR_SECONDS", 0.2, create=True), \
                mock.patch.object(sys, "stderr", log):
            t = threading.Thread(target=p._run_asr, daemon=True)
            t.start()
            p._asr_q.put(np.zeros(1600, dtype=np.float32))
            deadline = time.monotonic() + 3
            while not results and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.3)
            p.stop()
            t.join(2)
            log.seek(0)
            return log.read()

    def test_stuck_transcription_dumps_stacks_to_log(self):
        self.assertIn("_asr_loop", self.run_asr_once(0.6))

    def test_fast_transcription_writes_nothing(self):
        self.assertEqual(self.run_asr_once(0.0), "")

    def test_works_without_stderr(self):
        import pipeline as pipeline_mod

        results = []
        p = pipeline_mod.Pipeline(Settings(), on_result=lambda o, t: results.append(o))
        p._transcriber = mock.Mock()
        p._transcriber.transcribe.return_value = transcriber_mod.TranscriptionResult(text="hi", language="en")
        p._translator = mock.Mock()
        p._translator.translate.return_value = "selam"
        with mock.patch.object(sys, "stderr", None):
            workers = [threading.Thread(target=f, daemon=True) for f in (p._run_asr, p._run_mt)]
            for w in workers:
                w.start()
            p._asr_q.put(np.zeros(1600, dtype=np.float32))
            deadline = time.monotonic() + 3
            while not results and time.monotonic() < deadline:
                time.sleep(0.01)
            p.stop()
            for w in workers:
                w.join(2)
        self.assertEqual(results, ["hi"])


if __name__ == "__main__":
    unittest.main()
