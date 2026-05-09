import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.engine_vi import (
    VietnameseTTSEngine,
    SAMPLE_RATE,
    MODES,
    INSTALL_GUIDE,
    _segment_for_codeswitching,
)

VIENEU_AVAILABLE = VietnameseTTSEngine().is_available()

# Text samples used across tests
PURE_VI          = "Xin chào, tôi là trợ lý AI của bạn."
CODESWITCHED     = "Hệ thống dùng machine learning để phân tích dữ liệu."
TURBO_VI         = "Tôi muốn nghe giọng đọc nhanh hơn."
COMPLEX_TONES    = "Bầu trời xanh ngắt, cô bé ngã xuống rồi khóc nức nở."
# Covers: huyền (bầu), hỏi (ngắt), ngã (ngã, nở), nặng (ngắt end), sắc (xuống)


# ---------------------------------------------------------------------------
# Code-switching segmentation (no vieneu needed)
# ---------------------------------------------------------------------------

class TestCodeSwitching(unittest.TestCase):

    def test_pure_vietnamese_all_vi(self):
        segs = _segment_for_codeswitching("Xin chào các bạn")
        langs = {s["lang"] for s in segs}
        self.assertEqual(langs, {"vi"}, f"Expected all-vi, got segments: {segs}")

    def test_english_terms_tagged_en(self):
        segs = _segment_for_codeswitching("Hệ thống dùng machine learning để phân tích")
        en_texts = " ".join(s["text"] for s in segs if s["lang"] == "en")
        self.assertIn("machine", en_texts)
        self.assertIn("learning", en_texts)

    def test_ai_acronym_tagged_en(self):
        segs = _segment_for_codeswitching("Chúng tôi sử dụng AI trong sản phẩm")
        en_texts = " ".join(s["text"] for s in segs if s["lang"] == "en")
        self.assertIn("AI", en_texts)

    def test_vietnamese_with_full_tones_all_vi(self):
        segs = _segment_for_codeswitching(COMPLEX_TONES)
        # All segments that contain diacritic characters must be 'vi'
        for seg in segs:
            import re
            vi_chars = re.search(r"[àáảãạăắặẵẳặâấậẫẩèéẻẽẹêềếểễệ"
                                 r"ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụ"
                                 r"ưừứửữựỳýỷỹỵđĐ]", seg["text"])
            if vi_chars:
                self.assertEqual(seg["lang"], "vi",
                                 f"Segment with diacritics tagged as EN: {seg}")

    def test_segments_cover_full_text(self):
        text = "Hệ thống dùng deep learning và AI để nhận dạng giọng nói"
        segs = _segment_for_codeswitching(text)
        # All tokens from original text must appear somewhere in segments
        self.assertGreater(len(segs), 0)
        reconstructed = " ".join(s["text"] for s in segs)
        # At minimum every word root should appear
        for word in ["Hệ", "deep", "learning", "AI", "nhận"]:
            self.assertIn(word, reconstructed)

    def test_returns_list_of_dicts(self):
        segs = _segment_for_codeswitching("Xin chào")
        self.assertIsInstance(segs, list)
        for seg in segs:
            self.assertIn("lang", seg)
            self.assertIn("text", seg)
            self.assertIn(seg["lang"], {"vi", "en"})


# ---------------------------------------------------------------------------
# Engine interface (no vieneu needed)
# ---------------------------------------------------------------------------

class TestEngineInterface(unittest.TestCase):

    def setUp(self):
        self.engine = VietnameseTTSEngine()

    def test_list_modes_returns_standard_and_turbo(self):
        modes = self.engine.list_modes()
        self.assertEqual(set(modes), {"standard", "turbo"})

    def test_list_modes_is_list(self):
        self.assertIsInstance(self.engine.list_modes(), list)

    def test_sample_rate_is_24000(self):
        self.assertEqual(self.engine.sample_rate(), 24_000)

    def test_is_available_returns_bool(self):
        self.assertIsInstance(self.engine.is_available(), bool)


# ---------------------------------------------------------------------------
# Input validation (no vieneu needed — validation runs before import)
# ---------------------------------------------------------------------------

class TestInputValidation(unittest.TestCase):

    def setUp(self):
        self.engine = VietnameseTTSEngine()

    def _call(self, **kwargs):
        defaults = dict(text="Xin chào", speed=1.0, mode="standard")
        defaults.update(kwargs)
        return self.engine.synthesize(**defaults)

    def test_raises_on_empty_text(self):
        with self.assertRaises(ValueError):
            self._call(text="")

    def test_raises_on_whitespace_only(self):
        with self.assertRaises(ValueError):
            self._call(text="   \n  ")

    def test_raises_on_invalid_mode(self):
        with self.assertRaises(ValueError):
            self._call(mode="ultra")

    def test_raises_on_speed_too_low(self):
        with self.assertRaises(ValueError):
            self._call(speed=0.1)

    def test_raises_on_speed_too_high(self):
        with self.assertRaises(ValueError):
            self._call(speed=3.0)

    def test_raises_on_missing_ref_audio(self):
        with self.assertRaises(FileNotFoundError):
            self._call(ref_audio_path="/nonexistent/path/ref.wav")

    def test_raises_importerror_when_vieneu_missing(self):
        if VIENEU_AVAILABLE:
            self.skipTest("vieneu is installed — skip ImportError test")
        with self.assertRaises(ImportError) as ctx:
            self._call()
        self.assertIn("VieNeu-TTS", str(ctx.exception))

    def test_importerror_message_contains_install_hint(self):
        if VIENEU_AVAILABLE:
            self.skipTest("vieneu is installed — skip ImportError test")
        with self.assertRaises(ImportError) as ctx:
            self._call()
        msg = str(ctx.exception)
        self.assertTrue(
            "pip install" in msg or "github" in msg.lower(),
            f"ImportError missing install hint: {msg}"
        )


# ---------------------------------------------------------------------------
# Synthesis tests — skipped when vieneu not installed
# ---------------------------------------------------------------------------

@unittest.skipUnless(VIENEU_AVAILABLE, "vieneu not installed — skipping synthesis tests")
class TestSynthesis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = VietnameseTTSEngine()

    def test_pure_vietnamese(self):
        audio = self.engine.synthesize(PURE_VI)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio), 0)

    def test_codeswitched_vietnamese_english(self):
        audio = self.engine.synthesize(CODESWITCHED)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_turbo_mode(self):
        audio = self.engine.synthesize(TURBO_VI, mode="turbo")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_complex_tone_marks(self):
        # sắc, huyền, hỏi, ngã, nặng — all six tones present
        audio = self.engine.synthesize(COMPLEX_TONES)
        duration = len(audio) / SAMPLE_RATE
        self.assertGreater(duration, 1.0, "Audio too short for complex-tone sentence")

    def test_speed_variation(self):
        slow = self.engine.synthesize("Xin chào.", speed=0.8)
        fast = self.engine.synthesize("Xin chào.", speed=1.5)
        self.assertGreater(len(slow), len(fast),
                           "Slower speed should produce longer audio")

    def test_standard_mode_returns_float32(self):
        audio = self.engine.synthesize(PURE_VI, mode="standard")
        self.assertEqual(audio.dtype, np.float32)

    def test_model_cached_after_first_call(self):
        self.engine.synthesize(PURE_VI)
        id_before = id(self.engine._tts)
        self.engine.synthesize(PURE_VI)
        self.assertEqual(id_before, id(self.engine._tts),
                         "Model was reloaded — lazy cache broken")


if __name__ == "__main__":
    unittest.main(verbosity=2)
