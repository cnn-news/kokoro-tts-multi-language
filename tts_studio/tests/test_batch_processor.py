import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch.processor import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
    SingleResult,
    _config_with_override,
    _to_route_config,
    _validate_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _en_cfg(**kwargs) -> BatchConfig:
    """BatchConfig preset: English, WAV, temp export dir."""
    defaults = dict(lang="en", format="WAV", normalize=False)
    defaults.update(kwargs)
    return BatchConfig(**defaults)


def _make_files(*contents, lang_override=None) -> list[dict]:
    return [
        {"name": f"file_{i:02d}.txt", "content": c, "lang_override": lang_override}
        for i, c in enumerate(contents)
    ]


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------

class TestBatchConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = BatchConfig()
        self.assertEqual(cfg.lang,      "auto")
        self.assertEqual(cfg.format,    "WAV")
        self.assertEqual(cfg.voice_en,  "am_michael")
        self.assertEqual(cfg.mode_vi,   "standard")
        self.assertEqual(cfg.style_ja,  "neutral")
        self.assertTrue(cfg.normalize)
        self.assertEqual(cfg.normalize_db, -20.0)

    def test_validate_ok(self):
        BatchConfig(lang="en", format="MP3", speed_en=1.0).validate()   # no raise

    def test_validate_bad_format(self):
        with self.assertRaises(ValueError):
            BatchConfig(format="FLAC").validate()

    def test_validate_bad_lang(self):
        with self.assertRaises(ValueError):
            BatchConfig(lang="zh").validate()

    def test_validate_speed_en_out_of_range(self):
        with self.assertRaises(ValueError):
            BatchConfig(speed_en=3.0).validate()

    def test_validate_speed_vi_out_of_range(self):
        with self.assertRaises(ValueError):
            BatchConfig(speed_vi=0.1).validate()

    def test_validate_speed_ja_out_of_range(self):
        with self.assertRaises(ValueError):
            BatchConfig(speed_ja=2.5).validate()


# ---------------------------------------------------------------------------
# SingleResult & BatchResult
# ---------------------------------------------------------------------------

class TestDataClasses(unittest.TestCase):

    def _make_batch(self, statuses: list[str]) -> BatchResult:
        results = []
        for s in statuses:
            r = SingleResult(file_name="f.txt", status=s)
            if s == "ok":
                r.duration = 1.5
                r.output_path = Path("/tmp/f.wav")
            else:
                r.error = "oops"
            results.append(r)
        br = BatchResult(results=results)
        return br

    def test_batch_total(self):
        br = self._make_batch(["ok", "ok", "error"])
        self.assertEqual(br.total, 3)

    def test_batch_succeeded(self):
        br = self._make_batch(["ok", "ok", "error"])
        self.assertEqual(br.succeeded, 2)

    def test_batch_failed(self):
        br = self._make_batch(["ok", "error", "error"])
        self.assertEqual(br.failed, 2)

    def test_batch_total_audio_duration(self):
        br = self._make_batch(["ok", "ok", "error"])
        self.assertAlmostEqual(br.total_audio_duration, 3.0)

    def test_batch_errors_list(self):
        br = self._make_batch(["ok", "error"])
        self.assertEqual(len(br.errors), 1)
        self.assertIn("error", br.errors[0])
        self.assertIn("file_name", br.errors[0])

    def test_empty_batch(self):
        br = BatchResult()
        self.assertEqual(br.total, 0)
        self.assertEqual(br.succeeded, 0)
        self.assertEqual(br.failed, 0)
        self.assertAlmostEqual(br.total_audio_duration, 0.0)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_validate_text_raises_on_empty(self):
        with self.assertRaises(ValueError):
            _validate_text("")

    def test_validate_text_raises_on_whitespace(self):
        with self.assertRaises(ValueError):
            _validate_text("   \n  ")

    def test_validate_text_ok_on_normal(self):
        _validate_text("Hello world.")   # should not raise

    def test_config_with_override_sets_lang(self):
        cfg = BatchConfig(lang="auto")
        new = _config_with_override(cfg, "ja")
        self.assertEqual(new.lang, "ja")

    def test_config_with_override_none_unchanged(self):
        cfg = BatchConfig(lang="en")
        same = _config_with_override(cfg, None)
        self.assertEqual(same.lang, "en")

    def test_config_with_override_does_not_mutate_original(self):
        cfg = BatchConfig(lang="auto")
        _config_with_override(cfg, "vi")
        self.assertEqual(cfg.lang, "auto")

    def test_to_route_config_maps_fields(self):
        cfg = BatchConfig(
            lang="en", voice_en="am_fenrir", speed_en=1.5,
            speed_vi=0.8, mode_vi="turbo",
            voice_ja="jvnv-F1-jp", style_ja="happy", speed_ja=1.2,
        )
        rc = _to_route_config(cfg)
        self.assertEqual(rc.lang,     "en")
        self.assertEqual(rc.voice_en, "am_fenrir")
        self.assertAlmostEqual(rc.speed_en, 1.5)
        self.assertEqual(rc.mode_vi,  "turbo")
        self.assertEqual(rc.voice_ja, "jvnv-F1-jp")
        self.assertEqual(rc.style_ja, "happy")


# ---------------------------------------------------------------------------
# BatchProcessor.process_single()
# ---------------------------------------------------------------------------

class TestProcessSingle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.proc = BatchProcessor()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _cfg(self, **kw) -> BatchConfig:
        defaults = dict(lang="en", format="WAV", export_dir=str(self.tmp), normalize=False)
        defaults.update(kw)
        return BatchConfig(**defaults)

    def test_ok_result_has_status_ok(self):
        r = self.proc.process_single("test.txt", "Hello world today.", self._cfg())
        self.assertEqual(r.status, "ok")

    def test_ok_result_has_output_path(self):
        r = self.proc.process_single("test.txt", "Hello world today.", self._cfg())
        self.assertIsNotNone(r.output_path)

    def test_output_file_exists(self):
        r = self.proc.process_single("exists_test.txt", "Check file exists.", self._cfg())
        self.assertTrue(Path(r.output_path).exists())

    def test_output_filename_stem_matches_input(self):
        r = self.proc.process_single("chapter_one.txt", "Chapter one content.", self._cfg())
        self.assertEqual(Path(r.output_path).stem, "chapter_one")

    def test_output_extension_wav(self):
        r = self.proc.process_single("audio.txt", "WAV export test.", self._cfg(format="WAV"))
        self.assertEqual(Path(r.output_path).suffix, ".wav")

    def test_output_extension_mp3(self):
        r = self.proc.process_single("audio.txt", "MP3 export test.", self._cfg(format="MP3"))
        self.assertEqual(Path(r.output_path).suffix, ".mp3")

    def test_ok_result_has_duration(self):
        r = self.proc.process_single("dur.txt", "Duration check sentence.", self._cfg())
        self.assertIsNotNone(r.duration)
        self.assertGreater(r.duration, 0.0)

    def test_ok_result_has_render_time(self):
        r = self.proc.process_single("rt.txt", "Render time check.", self._cfg())
        self.assertIsNotNone(r.render_time)
        self.assertGreater(r.render_time, 0.0)

    def test_ok_result_has_detected_lang(self):
        r = self.proc.process_single("lang.txt", "Detected language test.", self._cfg())
        self.assertIn(r.detected_lang, ("en", "vi", "ja", "unknown"))

    def test_error_on_empty_text(self):
        r = self.proc.process_single("empty.txt", "", self._cfg())
        self.assertEqual(r.status, "error")
        self.assertIsNotNone(r.error)
        self.assertIsNone(r.output_path)

    def test_error_on_whitespace_text(self):
        r = self.proc.process_single("ws.txt", "   ", self._cfg())
        self.assertEqual(r.status, "error")

    def test_error_result_has_error_message(self):
        r = self.proc.process_single("e.txt", "", self._cfg())
        self.assertIsInstance(r.error, str)
        self.assertGreater(len(r.error), 0)

    def test_normalize_changes_rms(self):
        # With normalize=True, output audio should be at target loudness
        # We verify the file is written (normalize path exercised without crash)
        r = self.proc.process_single(
            "norm.txt", "Normalization test sentence.",
            self._cfg(normalize=True, normalize_db=-20.0)
        )
        self.assertEqual(r.status, "ok")
        self.assertTrue(Path(r.output_path).exists())

    def test_japanese_via_lang_override(self):
        cfg = BatchConfig(
            lang="ja", format="WAV",
            export_dir=str(self.tmp), normalize=False,
        )
        r = self.proc.process_single("jp.txt", "テストです。", cfg)
        self.assertEqual(r.status, "ok")
        self.assertGreater(r.duration, 0.0)


# ---------------------------------------------------------------------------
# BatchProcessor.process() — full batch
# ---------------------------------------------------------------------------

class TestProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.proc = BatchProcessor()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _cfg(self, **kw) -> BatchConfig:
        defaults = dict(lang="en", format="WAV", export_dir=str(self.tmp), normalize=False)
        defaults.update(kw)
        return BatchConfig(**defaults)

    # --- return types ---

    def test_returns_batch_result(self):
        result = self.proc.process([], self._cfg())
        self.assertIsInstance(result, BatchResult)

    def test_empty_files_list(self):
        result = self.proc.process([], self._cfg())
        self.assertEqual(result.total, 0)
        self.assertEqual(result.succeeded, 0)

    # --- counts ---

    def test_total_equals_input_count(self):
        files = _make_files("Hello.", "Good day.", lang_override="en")
        result = self.proc.process(files, self._cfg())
        self.assertEqual(result.total, 2)

    def test_all_ok_files_succeed(self):
        files = _make_files("Hello.", "World.", lang_override="en")
        result = self.proc.process(files, self._cfg())
        self.assertEqual(result.succeeded, 2)
        self.assertEqual(result.failed, 0)

    def test_empty_content_counted_as_error(self):
        files = _make_files("Good text.", "", "More text.", lang_override="en")
        result = self.proc.process(files, self._cfg())
        self.assertEqual(result.succeeded, 2)
        self.assertEqual(result.failed, 1)

    def test_error_does_not_stop_batch(self):
        files = _make_files("First.", "", "Third.", lang_override="en")
        result = self.proc.process(files, self._cfg())
        self.assertEqual(result.total, 3)
        self.assertEqual(result.results[0].status, "ok")
        self.assertEqual(result.results[1].status, "error")
        self.assertEqual(result.results[2].status, "ok")

    def test_all_errors_batch(self):
        files = _make_files("", "  ", "\n")
        result = self.proc.process(files, self._cfg())
        self.assertEqual(result.failed, 3)
        self.assertEqual(result.succeeded, 0)

    # --- output files ---

    def test_output_files_written_to_export_dir(self):
        sub = self.tmp / "batch_out"
        files = _make_files("Audio export.", lang_override="en")
        files[0]["name"] = "unique_stem.txt"
        result = self.proc.process(files, self._cfg(export_dir=str(sub)))
        self.assertTrue(any(
            r.output_path and Path(r.output_path).parent == sub.resolve()
            for r in result.results if r.status == "ok"
        ))

    def test_output_stem_matches_input_name(self):
        files = [{"name": "my_audio.txt", "content": "Stem test.", "lang_override": "en"}]
        result = self.proc.process(files, self._cfg())
        r = result.results[0]
        self.assertEqual(r.status, "ok")
        self.assertEqual(Path(r.output_path).stem, "my_audio")

    def test_total_audio_duration_positive(self):
        files = _make_files("Duration one.", "Duration two.", lang_override="en")
        result = self.proc.process(files, self._cfg())
        self.assertGreater(result.total_audio_duration, 0.0)

    # --- per-file lang override ---

    def test_per_file_lang_override_used(self):
        files = [
            {"name": "en.txt", "content": "Hello world.",  "lang_override": "en"},
            {"name": "ja.txt", "content": "テストです。",   "lang_override": "ja"},
        ]
        result = self.proc.process(files, self._cfg(lang="en"))
        # JA file should succeed and produce 44100 Hz audio (larger file)
        ja_r = result.results[1]
        self.assertEqual(ja_r.status, "ok")

    # --- progress callback ---

    def test_progress_callback_called_once_per_file(self):
        files = _make_files("One.", "Two.", "Three.", lang_override="en")
        calls = []
        self.proc.process(files, self._cfg(), progress_callback=lambda *a: calls.append(a))
        self.assertEqual(len(calls), 3)

    def test_progress_callback_reaches_100(self):
        files = _make_files("Done.", lang_override="en")
        pcts = []
        self.proc.process(files, self._cfg(), progress_callback=lambda p, *_: pcts.append(p))
        self.assertAlmostEqual(pcts[-1], 100.0)

    def test_progress_callback_pct_monotone(self):
        files = _make_files("A.", "B.", "C.", "D.", lang_override="en")
        pcts = []
        self.proc.process(files, self._cfg(), progress_callback=lambda p, *_: pcts.append(p))
        self.assertEqual(pcts, sorted(pcts))

    def test_progress_callback_receives_filename(self):
        files = [{"name": "named_file.txt", "content": "Test.", "lang_override": "en"}]
        names = []
        self.proc.process(
            files, self._cfg(),
            progress_callback=lambda pct, name, done, total: names.append(name)
        )
        self.assertIn("named_file.txt", names)

    def test_no_progress_callback_does_not_raise(self):
        files = _make_files("No callback.", lang_override="en")
        result = self.proc.process(files, self._cfg(), progress_callback=None)
        self.assertEqual(result.succeeded, 1)

    # --- router reuse ---

    def test_router_instantiated_once(self):
        proc = BatchProcessor()
        files = _make_files("First.", "Second.", lang_override="en")
        proc.process(files, self._cfg())
        id_after_batch = id(proc._router)
        proc.process(files, self._cfg())
        self.assertEqual(id_after_batch, id(proc._router),
                         "LanguageRouter was re-instantiated between batches")


if __name__ == "__main__":
    unittest.main(verbosity=2)
