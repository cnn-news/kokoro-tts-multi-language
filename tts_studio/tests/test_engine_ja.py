import sys
import time
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.engine_ja import (
    JapaneseTTSEngine,
    SAMPLE_RATE,
    VOICES,
    STYLES,
    STYLE_MAP,
)

# Text samples
HIRAGANA  = "こんにちは、げんきですか？"
KANJI     = "東京は日本の首都です。"
KATAKANA  = "コンピューターとインターネットが好きです。"
MIXED_JP  = "東京のコンビニでスマートフォンを買いました。"


class TestEngineInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = JapaneseTTSEngine()

    # ------------------------------------------------------------------
    # list_voices
    # ------------------------------------------------------------------

    def test_list_voices_returns_nonempty_dict(self):
        voices = self.engine.list_voices()
        self.assertIsInstance(voices, dict)
        self.assertGreater(len(voices), 0)

    def test_list_voices_contains_expected_ids(self):
        voices = self.engine.list_voices()
        for vid in ("jvnv-F1-jp", "jvnv-F2-jp", "jvnv-M1-jp", "jvnv-M2-jp"):
            self.assertIn(vid, voices, f"Missing voice: {vid}")

    def test_list_voices_labels_are_strings(self):
        for vid, label in self.engine.list_voices().items():
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    # ------------------------------------------------------------------
    # list_styles
    # ------------------------------------------------------------------

    def test_list_styles_returns_list(self):
        self.assertIsInstance(self.engine.list_styles(), list)

    def test_list_styles_contains_neutral_and_happy(self):
        styles = self.engine.list_styles()
        self.assertIn("neutral",   styles)
        self.assertIn("happy",     styles)
        self.assertIn("sad",       styles)
        self.assertIn("angry",     styles)
        self.assertIn("surprised", styles)

    def test_style_map_covers_all_listed_styles(self):
        for s in self.engine.list_styles():
            self.assertIn(s, STYLE_MAP,
                          f"Style {s!r} is in list_styles but missing from STYLE_MAP")

    # ------------------------------------------------------------------
    # sample_rate / device
    # ------------------------------------------------------------------

    def test_sample_rate_is_44100(self):
        self.assertEqual(self.engine.sample_rate(), 44_100)

    def test_device_is_string(self):
        self.assertIsInstance(self.engine.device(), str)
        self.assertIn(self.engine.device(), ("cpu", "cuda", "mps"))

    # ------------------------------------------------------------------
    # Validation (no heavy inference needed)
    # ------------------------------------------------------------------

    def test_raises_on_empty_text(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("", voice_id="jvnv-M1-jp", style="neutral", speed=1.0)

    def test_raises_on_whitespace_only(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("   ", voice_id="jvnv-M1-jp", style="neutral", speed=1.0)

    def test_raises_on_unknown_voice(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("テスト", voice_id="unknown-voice", style="neutral", speed=1.0)

    def test_raises_on_unknown_style(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("テスト", voice_id="jvnv-M1-jp", style="kawaii", speed=1.0)

    def test_raises_on_speed_too_low(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("テスト", voice_id="jvnv-M1-jp", style="neutral", speed=0.1)

    def test_raises_on_speed_too_high(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize("テスト", voice_id="jvnv-M1-jp", style="neutral", speed=3.0)


# ---------------------------------------------------------------------------
# Synthesis tests — require style-bert-vits2 and model download
# ---------------------------------------------------------------------------

class TestSynthesis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = JapaneseTTSEngine()

    # ------------------------------------------------------------------
    # Script types
    # ------------------------------------------------------------------

    def test_synthesize_hiragana(self):
        audio = self.engine.synthesize(HIRAGANA, voice_id="jvnv-M1-jp", style="neutral")
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertGreater(len(audio), 0)

    def test_synthesize_kanji(self):
        audio = self.engine.synthesize(KANJI, voice_id="jvnv-M1-jp", style="neutral")
        duration = len(audio) / SAMPLE_RATE
        self.assertGreater(duration, 0.5,  "Kanji sentence produced silent/short audio")
        self.assertLess(duration,    30.0, "Kanji sentence audio unexpectedly long")

    def test_synthesize_katakana_loanword(self):
        audio = self.engine.synthesize(KATAKANA, voice_id="jvnv-M1-jp", style="neutral")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_synthesize_mixed_kanji_katakana(self):
        audio = self.engine.synthesize(MIXED_JP, voice_id="jvnv-M1-jp", style="neutral")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def test_style_neutral(self):
        audio = self.engine.synthesize("テストです。", voice_id="jvnv-M1-jp", style="neutral")
        self.assertGreater(len(audio), 0)

    def test_style_happy(self):
        audio = self.engine.synthesize("やったー！最高です！", voice_id="jvnv-M1-jp", style="happy")
        self.assertGreater(len(audio), 0)

    def test_style_sad(self):
        audio = self.engine.synthesize("悲しいです。", voice_id="jvnv-M1-jp", style="sad")
        self.assertGreater(len(audio), 0)

    def test_style_angry(self):
        audio = self.engine.synthesize("怒っています！", voice_id="jvnv-M1-jp", style="angry")
        self.assertGreater(len(audio), 0)

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    def test_faster_speed_produces_shorter_audio(self):
        normal = self.engine.synthesize("テストです。", voice_id="jvnv-M1-jp", style="neutral", speed=1.0)
        fast   = self.engine.synthesize("テストです。", voice_id="jvnv-M1-jp", style="neutral", speed=1.5)
        self.assertGreater(len(normal), len(fast),
                           "Faster speed should produce shorter audio")

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    def test_female_voice_f1(self):
        audio = self.engine.synthesize("おはようございます。", voice_id="jvnv-F1-jp", style="neutral")
        self.assertGreater(len(audio), 0)

    def test_male_voice_m1(self):
        audio = self.engine.synthesize("こんばんは。", voice_id="jvnv-M1-jp", style="neutral")
        self.assertGreater(len(audio), 0)

    # ------------------------------------------------------------------
    # Model caching
    # ------------------------------------------------------------------

    def test_model_cached_after_first_call(self):
        self.engine.synthesize(HIRAGANA, voice_id="jvnv-M1-jp", style="neutral")
        id_before = id(self.engine._models.get("jvnv-M1-jp"))
        self.engine.synthesize(HIRAGANA, voice_id="jvnv-M1-jp", style="neutral")
        id_after  = id(self.engine._models.get("jvnv-M1-jp"))
        self.assertEqual(id_before, id_after, "Model was reloaded — lazy cache broken")

    def test_different_voices_have_separate_model_instances(self):
        self.engine.synthesize(HIRAGANA,  voice_id="jvnv-M1-jp", style="neutral")
        self.engine.synthesize("こんにちは。", voice_id="jvnv-F1-jp", style="neutral")
        self.assertIn("jvnv-M1-jp", self.engine._models)
        self.assertIn("jvnv-F1-jp", self.engine._models)
        self.assertIsNot(
            self.engine._models["jvnv-M1-jp"],
            self.engine._models["jvnv-F1-jp"],
        )

    # ------------------------------------------------------------------
    # Output format
    # ------------------------------------------------------------------

    def test_audio_is_float32(self):
        audio = self.engine.synthesize(HIRAGANA, voice_id="jvnv-M1-jp", style="neutral")
        self.assertEqual(audio.dtype, np.float32)

    def test_audio_is_1d(self):
        audio = self.engine.synthesize(HIRAGANA, voice_id="jvnv-M1-jp", style="neutral")
        self.assertEqual(audio.ndim, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
