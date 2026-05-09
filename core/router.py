"""
LanguageRouter — auto-detect ngôn ngữ và dispatch tới engine tương ứng.

Detection pipeline:
  1. Kiểm tra text có đủ dài không (>= MIN_DETECT_CHARS).
  2. Dùng lingua-language-detector (chính xác hơn cho VI/JA).
  3. Fallback sang langdetect nếu lingua trả None (ngôn ngữ ngoài tập hỗ trợ).
  4. Nếu vẫn không xác định được → "unknown".
"""

import numpy as np

MIN_DETECT_CHARS = 5   # text ngắn hơn ngưỡng này không đủ để detect

# Lingua Language objects — lazy import để tránh import cost ở module level
_LINGUA_DETECTOR = None


def _get_lingua_detector():
    """Singleton lingua detector (chỉ build một lần, tốn ~200 ms)."""
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        _LINGUA_DETECTOR = (
            LanguageDetectorBuilder
            .from_languages(Language.ENGLISH, Language.VIETNAMESE, Language.JAPANESE)
            .with_minimum_relative_distance(0.1)
            .build()
        )
    return _LINGUA_DETECTOR


def _lingua_lang_to_code(lingua_lang) -> str:
    """Map lingua Language enum → 'en' | 'vi' | 'ja' | 'unknown'."""
    if lingua_lang is None:
        return "unknown"
    name = lingua_lang.name   # e.g. 'ENGLISH', 'VIETNAMESE', 'JAPANESE'
    return {"ENGLISH": "en", "VIETNAMESE": "vi", "JAPANESE": "ja"}.get(name, "unknown")


def _fallback_langdetect(text: str) -> str:
    """Use langdetect as a second opinion when lingua returns unknown."""
    try:
        from langdetect import detect, LangDetectException
        code = detect(text)           # e.g. 'en', 'vi', 'ja', 'ko', ...
        return code if code in ("en", "vi", "ja") else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public detect function (also importable standalone)
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the dominant language of *text*.

    Returns:
        "en" | "vi" | "ja" | "unknown"

    Raises:
        ValueError: if text is shorter than MIN_DETECT_CHARS characters.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    if len(text.strip()) < MIN_DETECT_CHARS:
        raise ValueError(
            f"text is too short to detect language reliably "
            f"(need >= {MIN_DETECT_CHARS} characters, got {len(text.strip())})"
        )

    detector = _get_lingua_detector()
    result = detector.detect_language_of(text)
    code = _lingua_lang_to_code(result)

    # If lingua couldn't decide (None), ask langdetect
    if code == "unknown":
        code = _fallback_langdetect(text)

    return code


# ---------------------------------------------------------------------------
# RouteConfig — thin typed wrapper around a plain dict
# ---------------------------------------------------------------------------

class RouteConfig:
    """
    Holds per-engine parameters for a single synthesis request.

    All fields are optional; defaults mirror config.py defaults.
    """

    def __init__(
        self,
        lang: str = "auto",          # "auto" | "en" | "vi" | "ja"
        # English (Kokoro)
        voice_en: str = "am_michael",
        speed_en: float = 1.3,
        # Vietnamese (VieNeu)
        voice_vi: str = "Ly",        # preset voice ID — mặc định Trúc Ly
        speed_vi: float = 1.0,
        mode_vi: str = "standard",
        ref_audio_vi: str | None = None,
        # Japanese (Style-Bert-VITS2)
        voice_ja: str = "jvnv-M1-jp",
        style_ja: str = "neutral",
        speed_ja: float = 1.0,
    ):
        self.lang = lang
        self.voice_en = voice_en
        self.speed_en = speed_en
        self.voice_vi = voice_vi
        self.speed_vi = speed_vi
        self.mode_vi = mode_vi
        self.ref_audio_vi = ref_audio_vi
        self.voice_ja = voice_ja
        self.style_ja = style_ja
        self.speed_ja = speed_ja

    @classmethod
    def from_dict(cls, d: dict) -> "RouteConfig":
        """Create from a plain dict, ignoring unknown keys."""
        known = {
            "lang", "voice_en", "speed_en",
            "voice_vi", "speed_vi", "mode_vi", "ref_audio_vi",
            "voice_ja", "style_ja", "speed_ja",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# LanguageRouter
# ---------------------------------------------------------------------------

class LanguageRouter:
    """
    Detects the language of incoming text and dispatches to the correct
    TTS engine.  Engines are lazy-instantiated and cached.
    """

    def __init__(self):
        self._engine_en = None
        self._engine_vi = None
        self._engine_ja = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Proxy to module-level detect_language()."""
        return detect_language(text)

    def route(
        self,
        text: str,
        config: RouteConfig | dict | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize *text* with the appropriate engine.

        Args:
            text:   Input text (any supported language).
            config: RouteConfig (or plain dict) with synthesis parameters.
                    Pass ``lang`` to override auto-detection.

        Returns:
            (audio_numpy_float32, sample_rate)

        Raises:
            ValueError:  text too short, or lang="unknown" and no fallback.
            ImportError: required engine not installed.
        """
        if config is None:
            config = RouteConfig()
        elif isinstance(config, dict):
            config = RouteConfig.from_dict(config)

        lang = config.lang
        if lang == "auto":
            lang = detect_language(text)

        if lang == "unknown":
            raise ValueError(
                f"Could not detect a supported language (EN/VI/JA) for the given text. "
                f"Pass lang='en', 'vi', or 'ja' to override."
            )

        if lang == "en":
            return self._synthesize_en(text, config)
        elif lang == "vi":
            return self._synthesize_vi(text, config)
        elif lang == "ja":
            return self._synthesize_ja(text, config)
        else:
            raise ValueError(f"Unsupported language: {lang!r}. Use 'en', 'vi', or 'ja'.")

    def route_batch(
        self,
        file_list: list[dict],
        config: RouteConfig | dict | None = None,
    ) -> list[dict]:
        """
        Synthesize a list of text items sequentially (avoids OOM from
        parallel model loading).

        Args:
            file_list: list of dicts with keys:
                         name         (str)  — used for result label
                         content      (str)  — text to synthesize
                         lang_override(str|None) — optional "en"/"vi"/"ja"
            config:    Base RouteConfig shared across all files.
                       Per-file lang_override takes precedence over config.lang.

        Returns:
            list of dicts:
              {name, status, audio, sample_rate, detected_lang, error}
        """
        if config is None:
            config = RouteConfig()
        elif isinstance(config, dict):
            config = RouteConfig.from_dict(config)

        results = []
        for item in file_list:
            name    = item.get("name", "unnamed")
            content = item.get("content", "")
            override = item.get("lang_override")

            per_file = RouteConfig.from_dict(vars(config))
            if override:
                per_file.lang = override

            try:
                audio, sr = self.route(content, per_file)
                # Detect lang for reporting (may already be known via override)
                try:
                    detected = override or detect_language(content)
                except ValueError:
                    detected = override or "unknown"
                results.append({
                    "name": name,
                    "status": "ok",
                    "audio": audio,
                    "sample_rate": sr,
                    "detected_lang": detected,
                    "error": None,
                })
            except Exception as exc:
                results.append({
                    "name": name,
                    "status": "error",
                    "audio": None,
                    "sample_rate": None,
                    "detected_lang": None,
                    "error": str(exc),
                })

        return results

    # ------------------------------------------------------------------
    # Engine accessors (lazy init)
    # ------------------------------------------------------------------

    def _get_engine_en(self):
        if self._engine_en is None:
            from core.engine_en import EnglishTTSEngine
            self._engine_en = EnglishTTSEngine()
        return self._engine_en

    def _get_engine_vi(self):
        if self._engine_vi is None:
            from core.engine_vi import VietnameseTTSEngine
            self._engine_vi = VietnameseTTSEngine()
        return self._engine_vi

    def _get_engine_ja(self):
        if self._engine_ja is None:
            from core.engine_ja import JapaneseTTSEngine
            self._engine_ja = JapaneseTTSEngine()
        return self._engine_ja

    # ------------------------------------------------------------------
    # Per-language synthesize helpers
    # ------------------------------------------------------------------

    def _synthesize_en(self, text: str, cfg: RouteConfig) -> tuple[np.ndarray, int]:
        from core.engine_en import SAMPLE_RATE
        engine = self._get_engine_en()
        audio  = engine.synthesize(text, voice_id=cfg.voice_en, speed=cfg.speed_en)
        return audio, SAMPLE_RATE

    def _synthesize_vi(self, text: str, cfg: RouteConfig) -> tuple[np.ndarray, int]:
        from core.engine_vi import SAMPLE_RATE
        engine = self._get_engine_vi()
        audio  = engine.synthesize(
            text,
            speed=cfg.speed_vi,
            mode=cfg.mode_vi,
            voice_id=cfg.voice_vi if cfg.voice_vi else None,
            ref_audio_path=cfg.ref_audio_vi,
        )
        return audio, SAMPLE_RATE

    def _synthesize_ja(self, text: str, cfg: RouteConfig) -> tuple[np.ndarray, int]:
        from core.engine_ja import SAMPLE_RATE
        engine = self._get_engine_ja()
        audio  = engine.synthesize(
            text,
            voice_id=cfg.voice_ja,
            style=cfg.style_ja,
            speed=cfg.speed_ja,
        )
        return audio, SAMPLE_RATE
