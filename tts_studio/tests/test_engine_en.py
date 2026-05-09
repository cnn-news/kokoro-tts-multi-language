import sys
import time
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.engine_en import EnglishTTSEngine, SAMPLE_RATE

SHORT_TEXT = "Hello world. This is a short test."
LONG_TEXT  = " ".join(["The quick brown fox jumps over the lazy dog."] * 10)  # ~450 chars
MULTILINE  = "First line of text.\nSecond line follows.\nThird line ends here."


class TestEnglishTTSEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = EnglishTTSEngine()

    # ------------------------------------------------------------------
    # list_voices
    # ------------------------------------------------------------------

    def test_list_voices_returns_nonempty_dict(self):
        voices = self.engine.list_voices()
        self.assertIsInstance(voices, dict)
        self.assertGreater(len(voices), 0)

    def test_list_voices_values_are_valid_ids(self):
        for label, vid in self.engine.list_voices().items():
            self.assertIsInstance(label, str)
            self.assertTrue(vid.startswith(("am_", "af_", "bm_", "bf_")),
                            f"Unexpected voice id format: {vid}")

    # ------------------------------------------------------------------
    # synthesize — output shape & type
    # ------------------------------------------------------------------

    def test_synthesize_short_text_returns_numpy_array(self):
        audio = self.engine.synthesize(SHORT_TEXT, voice_id="am_michael", speed=1.3)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertGreater(len(audio), 0)

    def test_synthesize_short_text_reasonable_duration(self):
        audio = self.engine.synthesize(SHORT_TEXT, voice_id="am_michael", speed=1.3)
        duration = len(audio) / SAMPLE_RATE
        self.assertGreater(duration, 0.5,  "Audio too short — likely silent")
        self.assertLess(duration,    30.0, "Audio unexpectedly long for short text")

    def test_synthesize_long_text_no_error(self):
        audio = self.engine.synthesize(LONG_TEXT, voice_id="am_michael", speed=1.3)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_synthesize_multiline_text(self):
        audio = self.engine.synthesize(MULTILINE, voice_id="am_michael", speed=1.3)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    # ------------------------------------------------------------------
    # synthesize — render speed
    # ------------------------------------------------------------------

    def test_render_speed_under_10s_for_100_words_on_warm_pipeline(self):
        text_100w = " ".join(["word"] * 100)
        # warm up (pipeline already loaded from previous tests)
        t = time.time()
        self.engine.synthesize(text_100w, voice_id="am_michael", speed=1.3)
        elapsed = time.time() - t
        self.assertLess(elapsed, 10.0,
                        f"Render took {elapsed:.1f}s — too slow for 100 words")

    # ------------------------------------------------------------------
    # synthesize — different voices
    # ------------------------------------------------------------------

    def test_synthesize_us_male_voice(self):
        audio = self.engine.synthesize("Testing US male voice.", voice_id="am_fenrir", speed=1.0)
        self.assertGreater(len(audio), 0)

    def test_synthesize_uk_male_voice(self):
        audio = self.engine.synthesize("Testing UK male voice.", voice_id="bm_george", speed=1.0)
        self.assertGreater(len(audio), 0)

    def test_synthesize_female_voice(self):
        audio = self.engine.synthesize("Testing female voice.", voice_id="af_heart", speed=1.0)
        self.assertGreater(len(audio), 0)

    # ------------------------------------------------------------------
    # synthesize — edge cases & error handling
    # ------------------------------------------------------------------

    def test_synthesize_raises_on_empty_text(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("", voice_id="am_michael")

    def test_synthesize_raises_on_whitespace_only(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("   \n  ", voice_id="am_michael")

    # ------------------------------------------------------------------
    # Pipeline caching
    # ------------------------------------------------------------------

    def test_pipeline_cached_after_first_call(self):
        # Call synthesize twice — second call must reuse cached pipeline
        self.engine.synthesize("Cache test one.", voice_id="am_michael", speed=1.3)
        before = id(self.engine._pipelines.get("a"))
        self.engine.synthesize("Cache test two.", voice_id="am_michael", speed=1.3)
        after  = id(self.engine._pipelines.get("a"))
        self.assertEqual(before, after, "Pipeline object was recreated — cache broken")

    def test_us_and_uk_pipelines_are_separate(self):
        self.engine.synthesize("US voice test.", voice_id="am_michael", speed=1.3)
        self.engine.synthesize("UK voice test.", voice_id="bm_george",  speed=1.3)
        self.assertIn("a", self.engine._pipelines)
        self.assertIn("b", self.engine._pipelines)
        self.assertIsNot(self.engine._pipelines["a"], self.engine._pipelines["b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
