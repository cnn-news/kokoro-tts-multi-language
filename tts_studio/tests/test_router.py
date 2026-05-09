import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.router import (
    detect_language,
    LanguageRouter,
    RouteConfig,
    MIN_DETECT_CHARS,
)

# ---------------------------------------------------------------------------
# Module-level detect_language tests  (no TTS engine needed)
# ---------------------------------------------------------------------------

class TestDetectLanguage(unittest.TestCase):

    # --- Required cases from TASK-402 spec ---

    def test_english_detected(self):
        self.assertEqual(detect_language("Hello world, how are you today?"), "en")

    def test_vietnamese_detected(self):
        self.assertEqual(detect_language("Xin chào bạn, hôm nay thế nào?"), "vi")

    def test_japanese_detected(self):
        self.assertEqual(detect_language("こんにちは、元気ですか？"), "ja")

    def test_korean_returns_unknown(self):
        # Korean not in supported set → unknown
        self.assertEqual(detect_language("안녕하세요, 반갑습니다."), "unknown")

    def test_vietnamese_with_full_tones_not_misdetected_as_en(self):
        vi_text = "Hệ thống phân tích dữ liệu lớn và hiện đại."
        result = detect_language(vi_text)
        self.assertEqual(result, "vi",
                         f"Vietnamese with diacritics misdetected as {result!r}")

    # --- Additional accuracy tests ---

    def test_japanese_kanji(self):
        self.assertEqual(detect_language("東京は日本の首都です。"), "ja")

    def test_japanese_katakana(self):
        self.assertEqual(detect_language("コンピューターとインターネット。"), "ja")

    def test_english_longer_sentence(self):
        self.assertEqual(
            detect_language("The quick brown fox jumps over the lazy dog."), "en"
        )

    def test_vietnamese_no_misdetect_with_diacritics(self):
        # Sentence with all six Vietnamese tone marks present
        text = "Bầu trời xanh ngắt, chiếc xe đạp bị hỏng rồi."
        self.assertEqual(detect_language(text), "vi")

    def test_chinese_mapped_to_japanese(self):
        # Chinese shares CJK characters with Japanese.  Our 3-language detector
        # (EN/VI/JA) has no Chinese class, so Chinese text is mapped to the
        # nearest supported language — Japanese.  This is expected behaviour.
        result = detect_language("你好，今天天气怎么样？")
        self.assertIn(result, ("ja", "unknown"),
                      "Chinese should map to 'ja' or 'unknown', not EN/VI")

    # --- Edge cases ---

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            detect_language("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            detect_language("   ")

    def test_too_short_raises(self):
        short = "a" * (MIN_DETECT_CHARS - 1)
        with self.assertRaises(ValueError):
            detect_language(short)

    def test_exactly_min_length_does_not_raise(self):
        # MIN_DETECT_CHARS exactly — should attempt detection (may be unknown, not raise)
        text = "Hello"   # exactly 5 chars
        try:
            result = detect_language(text)
            self.assertIn(result, ("en", "vi", "ja", "unknown"))
        except ValueError:
            self.fail("detect_language raised ValueError for text of exactly MIN_DETECT_CHARS")

    def test_return_value_is_always_valid_code(self):
        samples = [
            "Good morning, have a nice day.",
            "Chúc bạn một ngày tốt lành.",
            "おはようございます。",
            "Привет, как дела?",   # Russian → unknown
        ]
        valid = {"en", "vi", "ja", "unknown"}
        for s in samples:
            result = detect_language(s)
            self.assertIn(result, valid, f"Invalid code {result!r} for {s!r}")


# ---------------------------------------------------------------------------
# RouteConfig tests
# ---------------------------------------------------------------------------

class TestRouteConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RouteConfig()
        self.assertEqual(cfg.lang, "auto")
        self.assertEqual(cfg.voice_en, "am_michael")
        self.assertEqual(cfg.mode_vi, "standard")
        self.assertEqual(cfg.style_ja, "neutral")

    def test_from_dict_picks_known_keys(self):
        cfg = RouteConfig.from_dict({"lang": "en", "speed_en": 1.5, "unknown_key": "x"})
        self.assertEqual(cfg.lang, "en")
        self.assertAlmostEqual(cfg.speed_en, 1.5)

    def test_from_dict_ignores_unknown_keys(self):
        # Should not raise
        cfg = RouteConfig.from_dict({"lang": "ja", "foo": "bar", "baz": 123})
        self.assertEqual(cfg.lang, "ja")

    def test_lang_override_stored(self):
        cfg = RouteConfig(lang="vi")
        self.assertEqual(cfg.lang, "vi")


# ---------------------------------------------------------------------------
# LanguageRouter — detection proxy
# ---------------------------------------------------------------------------

class TestLanguageRouterDetect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = LanguageRouter()

    def test_detect_via_router_english(self):
        self.assertEqual(self.router.detect_language("Hello world, nice to meet you."), "en")

    def test_detect_via_router_vietnamese(self):
        self.assertEqual(self.router.detect_language("Xin chào, tôi đến từ Việt Nam."), "vi")

    def test_detect_via_router_japanese(self):
        self.assertEqual(self.router.detect_language("日本語のテストです。"), "ja")

    def test_detect_via_router_unknown(self):
        self.assertEqual(self.router.detect_language("안녕하세요, 감사합니다."), "unknown")


# ---------------------------------------------------------------------------
# LanguageRouter — route() validation (no TTS engine needed)
# ---------------------------------------------------------------------------

class TestRouterValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = LanguageRouter()

    def test_route_unknown_lang_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.router.route("안녕하세요, 반갑습니다.", RouteConfig())

    def test_route_unsupported_lang_override_raises(self):
        with self.assertRaises(ValueError):
            self.router.route("some text here please", RouteConfig(lang="ko"))

    def test_route_too_short_text_raises(self):
        with self.assertRaises(ValueError):
            self.router.route("hi", RouteConfig())

    def test_route_empty_text_raises(self):
        with self.assertRaises(ValueError):
            self.router.route("", RouteConfig())

    def test_route_config_from_dict(self):
        # Verify from_dict plumbing reaches route without raising a config error
        cfg = RouteConfig.from_dict({"lang": "en", "voice_en": "am_michael", "speed_en": 1.0})
        self.assertEqual(cfg.lang, "en")


# ---------------------------------------------------------------------------
# LanguageRouter — route() end-to-end synthesis
# ---------------------------------------------------------------------------

class TestRouterSynthesis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = LanguageRouter()

    def _assert_audio(self, audio, sr, label=""):
        self.assertIsInstance(audio, np.ndarray, f"{label}: not ndarray")
        self.assertEqual(audio.dtype, np.float32, f"{label}: not float32")
        self.assertEqual(audio.ndim, 1, f"{label}: not 1D")
        self.assertGreater(len(audio), 0, f"{label}: empty audio")
        self.assertGreater(sr, 0, f"{label}: invalid sample rate")

    def test_route_auto_english(self):
        audio, sr = self.router.route(
            "Hello, this is an automatic routing test.",
            RouteConfig()
        )
        self._assert_audio(audio, sr, "auto EN")
        self.assertEqual(sr, 24_000)

    def test_route_auto_japanese(self):
        audio, sr = self.router.route(
            "自動ルーティングのテストです。",
            RouteConfig()
        )
        self._assert_audio(audio, sr, "auto JA")
        self.assertEqual(sr, 44_100)

    def test_route_override_english(self):
        audio, sr = self.router.route(
            "Override language to English.",
            RouteConfig(lang="en")
        )
        self._assert_audio(audio, sr, "override EN")

    def test_route_override_japanese(self):
        audio, sr = self.router.route(
            "テストです。",
            RouteConfig(lang="ja")
        )
        self._assert_audio(audio, sr, "override JA")

    def test_route_returns_correct_sample_rate_for_en(self):
        _, sr = self.router.route(
            "Sample rate check for English.",
            RouteConfig(lang="en")
        )
        self.assertEqual(sr, 24_000)

    def test_route_returns_correct_sample_rate_for_ja(self):
        _, sr = self.router.route(
            "サンプルレートのテスト。",
            RouteConfig(lang="ja")
        )
        self.assertEqual(sr, 44_100)

    def test_engine_en_reused_across_calls(self):
        self.router.route("First call.", RouteConfig(lang="en"))
        id_before = id(self.router._engine_en)
        self.router.route("Second call.", RouteConfig(lang="en"))
        self.assertEqual(id_before, id(self.router._engine_en),
                         "EnglishTTSEngine was re-instantiated — lazy cache broken")

    def test_engine_ja_reused_across_calls(self):
        self.router.route("最初の呼び出し。", RouteConfig(lang="ja"))
        id_before = id(self.router._engine_ja)
        self.router.route("二回目の呼び出し。", RouteConfig(lang="ja"))
        self.assertEqual(id_before, id(self.router._engine_ja),
                         "JapaneseTTSEngine was re-instantiated — lazy cache broken")


# ---------------------------------------------------------------------------
# LanguageRouter — route_batch()
# ---------------------------------------------------------------------------

class TestRouterBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = LanguageRouter()

    def test_batch_returns_result_for_each_item(self):
        batch = [
            {"name": "a.txt", "content": "Hello world today.",    "lang_override": "en"},
            {"name": "b.txt", "content": "テストファイルです。",   "lang_override": "ja"},
        ]
        results = self.router.route_batch(batch, RouteConfig())
        self.assertEqual(len(results), 2)

    def test_batch_ok_items_have_audio(self):
        batch = [
            {"name": "en.txt", "content": "Batch synthesis test.", "lang_override": "en"},
        ]
        results = self.router.route_batch(batch, RouteConfig())
        r = results[0]
        self.assertEqual(r["status"], "ok")
        self.assertIsInstance(r["audio"], np.ndarray)
        self.assertGreater(len(r["audio"]), 0)

    def test_batch_error_item_does_not_stop_batch(self):
        batch = [
            {"name": "ok.txt",  "content": "Good text here.",  "lang_override": "en"},
            {"name": "bad.txt", "content": "xyz",               "lang_override": None},
            {"name": "ok2.txt", "content": "More good text.",   "lang_override": "en"},
        ]
        results = self.router.route_batch(batch, RouteConfig())
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["status"], "error")
        self.assertEqual(results[2]["status"], "ok")

    def test_batch_error_item_has_error_message(self):
        batch = [{"name": "short.txt", "content": "x", "lang_override": None}]
        results = self.router.route_batch(batch, RouteConfig())
        self.assertEqual(results[0]["status"], "error")
        self.assertIsNotNone(results[0]["error"])
        self.assertIsInstance(results[0]["error"], str)

    def test_batch_per_file_lang_override(self):
        batch = [
            {"name": "ja.txt", "content": "Override test.", "lang_override": "ja"},
        ]
        results = self.router.route_batch(batch, RouteConfig(lang="en"))
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["sample_rate"], 44_100)   # JA, not EN

    def test_batch_empty_list(self):
        results = self.router.route_batch([], RouteConfig())
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
