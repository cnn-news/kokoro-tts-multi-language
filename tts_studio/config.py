"""
TTS Studio Pro — Configuration

Defaults được định nghĩa trong class Config.
Khi file config.json tồn tại cạnh config.py, các trường trong đó sẽ
override defaults tương ứng.

Usage:
    from config import Config
    cfg = Config.load()        # loads + merges config.json if present
    print(cfg.PORT)            # 8766
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

_HERE = Path(__file__).parent
_CONFIG_JSON = _HERE / "config.json"


@dataclass
class Config:
    # ── Server ────────────────────────────────────────────────────────────
    PORT: int = 8766

    # ── Language default ──────────────────────────────────────────────────
    DEFAULT_LANG: str = "auto"      # "auto" | "en" | "vi" | "ja"

    # ── English — Kokoro TTS ──────────────────────────────────────────────
    EN_DEFAULT_VOICE: str   = "am_michael"
    EN_DEFAULT_SPEED: float = 1.30
    EN_SAMPLE_RATE: int     = 24_000

    # ── Vietnamese — VieNeu-TTS ───────────────────────────────────────────
    VI_DEFAULT_MODE: str    = "standard"   # "standard" | "turbo"
    VI_DEFAULT_SPEED: float = 1.0
    VI_SAMPLE_RATE: int     = 24_000
    VI_REF_AUDIO_DIR: str   = "./voices/vi_reference/"

    # ── Japanese — Style-Bert-VITS2 ───────────────────────────────────────
    JA_DEFAULT_VOICE: str   = "jvnv-F1-jp"
    JA_DEFAULT_STYLE: str   = "neutral"
    JA_DEFAULT_SPEED: float = 1.0
    JA_SAMPLE_RATE: int     = 44_100

    # ── Audio export ──────────────────────────────────────────────────────
    DEFAULT_FORMAT: str     = "WAV"        # "WAV" | "MP3"
    MP3_BITRATE: str        = "192k"
    DEFAULT_EXPORT: str     = "./output/"

    # ── UI ────────────────────────────────────────────────────────────────
    OPEN_BROWSER: bool      = True

    # ── Performance ───────────────────────────────────────────────────────
    TEXT_WARN_CHARS: int    = 2000   # warn if single text exceeds this length
    LOG_RENDER_TIME: bool   = True   # print render time per file to stderr

    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> "Config":
        """
        Create a Config instance with defaults, then override from a JSON file.

        Args:
            config_path: Path to a JSON config file.  If None, looks for
                         config.json next to this module file.

        Returns:
            Config instance with merged values.
        """
        cfg = cls()
        path = Path(config_path) if config_path else _CONFIG_JSON

        if path.exists():
            try:
                overrides = json.loads(path.read_text(encoding="utf-8"))
                cfg._apply(overrides, source=str(path))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[config] Warning: could not load {path}: {exc}",
                      file=sys.stderr)

        return cfg

    def save(self, config_path: Path | str | None = None) -> Path:
        """
        Persist the current config to a JSON file (for user customisation).

        Returns the path that was written.
        """
        path = Path(config_path) if config_path else _CONFIG_JSON
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _apply(self, overrides: dict, source: str = "") -> None:
        """Override fields from a dict, ignoring unknown keys."""
        known = {f for f in self.__dataclass_fields__}
        for key, value in overrides.items():
            if key in known:
                expected_type = type(getattr(self, key))
                try:
                    setattr(self, key, expected_type(value))
                except (TypeError, ValueError) as exc:
                    print(
                        f"[config] Warning: cannot apply {key}={value!r} "
                        f"from {source}: {exc}",
                        file=sys.stderr,
                    )
            else:
                pass  # silently ignore unknown keys — forward compat

    def to_batch_config_kwargs(self) -> dict:
        """Return a dict suitable for BatchConfig(**...) construction."""
        return {
            "lang":         self.DEFAULT_LANG,
            "voice_en":     self.EN_DEFAULT_VOICE,
            "speed_en":     self.EN_DEFAULT_SPEED,
            "speed_vi":     self.VI_DEFAULT_SPEED,
            "mode_vi":      self.VI_DEFAULT_MODE,
            "voice_ja":     self.JA_DEFAULT_VOICE,
            "style_ja":     self.JA_DEFAULT_STYLE,
            "speed_ja":     self.JA_DEFAULT_SPEED,
            "format":       self.DEFAULT_FORMAT,
            "export_dir":   self.DEFAULT_EXPORT,
            "bitrate":      self.MP3_BITRATE,
        }

    def to_route_config_kwargs(self) -> dict:
        """Return a dict suitable for RouteConfig(**...) construction."""
        return {
            "lang":     self.DEFAULT_LANG,
            "voice_en": self.EN_DEFAULT_VOICE,
            "speed_en": self.EN_DEFAULT_SPEED,
            "speed_vi": self.VI_DEFAULT_SPEED,
            "mode_vi":  self.VI_DEFAULT_MODE,
            "voice_ja": self.JA_DEFAULT_VOICE,
            "style_ja": self.JA_DEFAULT_STYLE,
            "speed_ja": self.JA_DEFAULT_SPEED,
        }
