"""
BatchProcessor — xử lý hàng loạt file .txt thành audio WAV/MP3.

Chạy tuần tự (sequential) để tránh OOM khi nhiều TTS model load cùng lúc.
Progress được thông báo qua callback sau mỗi file.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_TEXT_WARN_CHARS = 2000   # warn if single file exceeds this

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """
    Cấu hình cho một batch synthesis job.

    Synthesis params tương ứng với RouteConfig; export params chỉ dùng ở đây.
    """
    # --- Language ---
    lang: str = "auto"           # "auto" | "en" | "vi" | "ja"

    # --- English (Kokoro) ---
    voice_en: str  = "am_michael"
    speed_en: float = 1.3

    # --- Vietnamese (VieNeu) ---
    speed_vi: float  = 1.0
    mode_vi: str     = "standard"
    ref_audio_vi: str | None = None

    # --- Japanese (Style-Bert-VITS2) ---
    voice_ja: str  = "jvnv-M1-jp"
    style_ja: str  = "neutral"
    speed_ja: float = 1.0

    # --- Export ---
    format: str      = "WAV"      # "WAV" | "MP3"
    export_dir: str  = "./output"
    bitrate: str     = "192k"     # MP3 bitrate

    # --- Post-processing ---
    normalize: bool  = True
    normalize_db: float = -20.0

    def validate(self) -> None:
        if self.format not in ("WAV", "MP3"):
            raise ValueError(f"format must be 'WAV' or 'MP3', got {self.format!r}")
        if not (0.5 <= self.speed_en <= 2.0):
            raise ValueError(f"speed_en must be in [0.5, 2.0], got {self.speed_en}")
        if not (0.5 <= self.speed_vi <= 2.0):
            raise ValueError(f"speed_vi must be in [0.5, 2.0], got {self.speed_vi}")
        if not (0.5 <= self.speed_ja <= 2.0):
            raise ValueError(f"speed_ja must be in [0.5, 2.0], got {self.speed_ja}")
        if self.lang not in ("auto", "en", "vi", "ja"):
            raise ValueError(f"lang must be 'auto'/'en'/'vi'/'ja', got {self.lang!r}")


@dataclass
class SingleResult:
    """Kết quả tổng hợp của một file."""
    file_name: str
    status: str                  # "ok" | "error"
    output_path: Path | None   = None
    duration: float | None     = None   # audio duration in seconds
    detected_lang: str | None  = None
    render_time: float | None  = None   # wall-clock synthesis time
    error: str | None          = None


@dataclass
class BatchResult:
    """Tổng hợp kết quả toàn bộ batch job."""
    results: list[SingleResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def total_audio_duration(self) -> float:
        """Sum of successfully synthesized audio durations (seconds)."""
        return sum(r.duration or 0.0 for r in self.results if r.status == "ok")

    @property
    def errors(self) -> list[dict]:
        """List of {file_name, error} for failed items."""
        return [
            {"file_name": r.file_name, "error": r.error}
            for r in self.results if r.status == "error"
        ]


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

# Type alias for progress callback:
#   callback(pct, current_file, completed, total)
ProgressCallback = Callable[[float, str, int, int], None]


class BatchProcessor:
    """
    Processes a list of text items to audio files sequentially.

    Engines are lazy-loaded through LanguageRouter on the first request and
    cached for all subsequent files in the same batch.
    """

    def __init__(self) -> None:
        self._router = None     # LanguageRouter, lazy-init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        files: list[dict],
        config: BatchConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchResult:
        """
        Synthesize a list of text items to audio files.

        Args:
            files:    List of dicts with keys:
                        name         (str)       — used as output filename stem
                        content      (str)       — text to synthesize
                        lang_override(str|None)  — force language ("en"/"vi"/"ja")
            config:   BatchConfig with synthesis and export settings.
            progress_callback:
                      Called after each file:
                        callback(pct_done, current_file_name, n_completed, n_total)

        Returns:
            BatchResult with per-file SingleResult entries.
        """
        if config is None:
            config = BatchConfig()
        config.validate()

        export_dir = _prepare_export_dir(config.export_dir)
        batch_result = BatchResult()
        total = len(files)

        for idx, item in enumerate(files):
            name     = item.get("name", f"file_{idx:03d}")
            content  = item.get("content", "")
            override = item.get("lang_override")

            # Build per-file config with possible lang override
            file_config = _config_with_override(config, override)

            result = self.process_single(name, content, file_config, export_dir)
            batch_result.results.append(result)

            # Log render time to console
            if result.status == "ok":
                print(
                    f"[batch] {idx+1}/{total}  ✓  {name}  "
                    f"{result.duration:.2f}s audio  "
                    f"({result.render_time:.1f}s render)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[batch] {idx+1}/{total}  ✗  {name}  "
                    f"ERROR: {result.error}",
                    file=sys.stderr,
                )

            if progress_callback is not None:
                completed = idx + 1
                pct = completed / total * 100.0
                progress_callback(pct, name, completed, total)

        return batch_result

    def process_single(
        self,
        name: str,
        text: str,
        config: BatchConfig,
        export_dir: Path | None = None,
    ) -> SingleResult:
        """
        Synthesize a single text item and write the output file.

        Args:
            name:       Logical name; the stem is used for the output filename.
            text:       Text to synthesize.
            config:     BatchConfig controlling synthesis and export.
            export_dir: Override export directory (optional; falls back to config).

        Returns:
            SingleResult with status, output_path, duration, etc.
        """
        out_dir = export_dir or _prepare_export_dir(config.export_dir)
        t_start = time.perf_counter()

        try:
            _validate_text(text)
            _warn_long_text(name, text)

            from core.router import LanguageRouter, RouteConfig, detect_language

            router    = self._get_router()
            route_cfg = _to_route_config(config)

            # Synthesize
            audio, sr = router.route(text, route_cfg)

            # Detect language for the result record
            try:
                detected_lang = detect_language(text)
            except ValueError:
                detected_lang = config.lang if config.lang != "auto" else "unknown"

            # Optional normalization
            if config.normalize:
                from core.audio_utils import normalize_audio
                audio = normalize_audio(audio, target_db=config.normalize_db)

            # Export
            stem = Path(name).stem or name
            output_path = _export_audio(audio, sr, stem, out_dir, config)
            duration    = len(audio) / sr

            return SingleResult(
                file_name    = name,
                status       = "ok",
                output_path  = output_path,
                duration     = duration,
                detected_lang= detected_lang,
                render_time  = time.perf_counter() - t_start,
            )

        except Exception as exc:
            return SingleResult(
                file_name   = name,
                status      = "error",
                render_time = time.perf_counter() - t_start,
                error       = str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_router(self):
        if self._router is None:
            from core.router import LanguageRouter
            self._router = LanguageRouter()
        return self._router


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------

def _prepare_export_dir(export_dir: str | Path) -> Path:
    from core.audio_utils import validate_output_dir
    return validate_output_dir(export_dir)


def _validate_text(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("content is empty — nothing to synthesize")


def _warn_long_text(name: str, text: str) -> None:
    if len(text) > _TEXT_WARN_CHARS:
        print(
            f"[batch] WARNING: '{name}' is {len(text)} chars "
            f"(> {_TEXT_WARN_CHARS}) — synthesis may be slow on CPU.",
            file=sys.stderr,
        )


def _config_with_override(config: BatchConfig, lang_override: str | None) -> BatchConfig:
    """Return a copy of config with lang set to lang_override if provided."""
    if lang_override is None:
        return config
    import dataclasses
    c = dataclasses.replace(config, lang=lang_override)
    return c


def _to_route_config(cfg: BatchConfig):
    from core.router import RouteConfig
    return RouteConfig(
        lang         = cfg.lang,
        voice_en     = cfg.voice_en,
        speed_en     = cfg.speed_en,
        speed_vi     = cfg.speed_vi,
        mode_vi      = cfg.mode_vi,
        ref_audio_vi = cfg.ref_audio_vi,
        voice_ja     = cfg.voice_ja,
        style_ja     = cfg.style_ja,
        speed_ja     = cfg.speed_ja,
    )


def _export_audio(
    audio,
    sr: int,
    stem: str,
    out_dir: Path,
    config: BatchConfig,
) -> Path:
    from core.audio_utils import export_wav, export_mp3

    if config.format == "MP3":
        path = out_dir / f"{stem}.mp3"
        return export_mp3(audio, sr, path, bitrate=config.bitrate)
    else:
        path = out_dir / f"{stem}.wav"
        return export_wav(audio, sr, path)
