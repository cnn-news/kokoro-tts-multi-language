"""
TTS Studio Pro — HTTP Server
Port mặc định: 8766
Dùng http.server built-in — không cần Flask/FastAPI.

Endpoints:
  GET  /                    → Serve ui/index.html
  GET  /download?path=...   → Download audio file
  GET  /api/job/<job_id>    → Poll batch job status  (TASK-702)

  POST /api/detect_lang     → {text} → {lang}
  POST /api/voices          → {lang} → {voices:[{label,id}]}
  POST /api/synthesize      → {text,lang,voice,speed,format} → {audio_b64,duration}
  POST /api/batch           → {files,voice_config,format,export_dir} → {job_id}
"""

import base64
import io
import json
import mimetypes
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8766

# ---------------------------------------------------------------------------
# JobStore — thread-safe batch job tracker  (TASK-702)
# ---------------------------------------------------------------------------

class JobStore:
    """Thread-safe dictionary of batch job states."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, job_id: str) -> dict:
        state = {
            "status":       "queued",   # queued | running | done | error
            "progress_pct": 0.0,
            "current_file": None,
            "completed":    0,
            "total":        0,
            "results":      [],
            "errors":       [],
            "started_at":   time.time(),
        }
        with self._lock:
            self._jobs[job_id] = state
        return state

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def append_result(self, job_id: str, result: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["results"].append(result)
                if result.get("status") == "error":
                    self._jobs[job_id]["errors"].append(result)


# ---------------------------------------------------------------------------
# Module-level singletons (set by start_server before first request)
# ---------------------------------------------------------------------------

_router    = None       # LanguageRouter — lazy-init on first synthesis
_router_lock = threading.Lock()   # one TTS call at a time
_job_store = JobStore()
_ui_dir    = Path(__file__).parent.parent / "ui"
_export_dir= Path(__file__).parent.parent / "output"


def _get_router():
    global _router
    if _router is None:
        from core.router import LanguageRouter
        _router = LanguageRouter()
    return _router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data: dict) -> dict:
    return {"status": "ok", **data}


def _err(message: str, fix: str | None = None) -> dict:
    out = {"status": "error", "message": message}
    if fix:
        out["fix"] = fix
    return out


def _read_body(handler: "TTSRequestHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw    = handler.rfile.read(length)
    return json.loads(raw) if raw else {}


def _send_json(handler: "TTSRequestHandler", code: int, body: dict) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type",   "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def _audio_to_b64_wav(audio_np, sample_rate: int) -> str:
    """Encode numpy float32 audio as base64 WAV string."""
    import soundfile as sf
    import numpy as np
    buf = io.BytesIO()
    sf.write(buf, audio_np.astype(np.float32), sample_rate,
             format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_download_path(raw_path: str) -> Path | None:
    """
    Resolve the requested download path and verify it is safe:
      - must exist
      - must be a file (not a directory or symlink outside tree)
      - must have an audio extension
    """
    try:
        p = Path(raw_path).resolve()
    except Exception:
        return None
    if not p.exists() or not p.is_file():
        return None
    if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
        return None
    return p


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class TTSRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default per-request logging
        pass

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._serve_ui()
        elif path == "/download":
            self._handle_download(parsed.query)
        elif path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            self._handle_job_status(job_id)
        else:
            _send_json(self, 404, _err(f"Not found: {path}"))

    # ----------------------------------------------------------------- POST

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        try:
            body = _read_body(self)
        except (json.JSONDecodeError, ValueError) as exc:
            _send_json(self, 400, _err(f"Invalid JSON body: {exc}"))
            return

        routes = {
            "/api/detect_lang": self._handle_detect_lang,
            "/api/voices":      self._handle_voices,
            "/api/synthesize":  self._handle_synthesize,
            "/api/batch":       self._handle_batch,
        }

        handler_fn = routes.get(path)
        if handler_fn is None:
            _send_json(self, 404, _err(f"Unknown endpoint: {path}"))
        else:
            try:
                handler_fn(body)
            except Exception as exc:
                traceback.print_exc()
                _send_json(self, 500, _err(str(exc)))

    # ---------------------------------------------------------------- CORS

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # --------------------------------------------------------- GET handlers

    def _serve_ui(self):
        html_file = _ui_dir / "index.html"
        if not html_file.exists():
            _send_json(self, 404, _err("UI not found. Run a build or check ui/index.html."))
            return
        content = html_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_download(self, query_string: str):
        params   = parse_qs(query_string)
        raw_path = params.get("path", [None])[0]
        if not raw_path:
            _send_json(self, 400, _err("Missing 'path' query parameter."))
            return

        file_path = _safe_download_path(raw_path)
        if file_path is None:
            _send_json(self, 404, _err(
                "File not found or not a valid audio file.",
                fix="Ensure the path points to an existing .wav or .mp3 file."
            ))
            return

        content  = file_path.read_bytes()
        mime, _  = mimetypes.guess_type(str(file_path))
        mime     = mime or "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type",        mime)
        self.send_header("Content-Length",      str(len(content)))
        self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _handle_job_status(self, job_id: str):
        """TASK-702: poll a running/completed batch job."""
        job = _job_store.get(job_id)
        if job is None:
            _send_json(self, 404, _err(f"Job not found: {job_id}"))
            return
        _send_json(self, 200, _ok(job))

    # -------------------------------------------------------- POST handlers

    def _handle_detect_lang(self, body: dict):
        text = body.get("text", "")
        if not text or not text.strip():
            _send_json(self, 400, _err("'text' field is required and must not be empty."))
            return

        try:
            from core.router import detect_language
            lang = detect_language(text)
            _send_json(self, 200, _ok({"lang": lang}))
        except ValueError as exc:
            _send_json(self, 400, _err(str(exc)))
        except ImportError as exc:
            _send_json(self, 503, _err(
                str(exc),
                fix="Run: pip install lingua-language-detector>=2.0.0"
            ))

    def _handle_voices(self, body: dict):
        lang = body.get("lang", "en").lower()

        try:
            if lang == "en":
                from core.engine_en import EnglishTTSEngine
                voices = EnglishTTSEngine().list_voices_grouped()
            elif lang == "vi":
                from core.engine_vi import _has_vieneu, VOICE_CATALOG
                if not _has_vieneu():
                    _send_json(self, 503, _err(
                        "VieNeu-TTS chưa được cài đặt",
                        fix="pip install vieneu",
                    ))
                    return
                voices = VOICE_CATALOG
            elif lang == "ja":
                from core.engine_ja import JapaneseTTSEngine
                voices = JapaneseTTSEngine().list_voices_grouped()
            else:
                _send_json(self, 400, _err(
                    f"Unsupported lang: {lang!r}. Use 'en', 'vi', or 'ja'."
                ))
                return
        except ImportError as exc:
            _send_json(self, 503, _err(
                f"Engine for language '{lang}' is not installed: {exc}",
                fix="Run python setup.py to install missing engines."
            ))
            return

        _send_json(self, 200, _ok({"voices": voices}))

    def _handle_synthesize(self, body: dict):
        text   = body.get("text",   "")
        lang   = body.get("lang",   "auto")
        voice  = body.get("voice",  None)
        fmt    = body.get("format", "wav").upper()

        # -- Validation --
        if not text or not text.strip():
            _send_json(self, 400, _err("'text' field is required and must not be empty."))
            return
        if fmt not in ("WAV", "MP3"):
            _send_json(self, 400, _err(
                "'format' must be 'wav' or 'mp3'.",
                fix="Set format to 'wav' (recommended) or 'mp3'."
            ))
            return

        try:
            speed = float(body.get("speed", 1.0))
        except (TypeError, ValueError):
            _send_json(self, 400, _err("'speed' must be a number between 0.5 and 2.0."))
            return
        if not (0.5 <= speed <= 2.0):
            _send_json(self, 400, _err(
                f"'speed' must be between 0.5 and 2.0 (got {speed}).",
                fix="Use speed=1.0 for normal pace, 1.3 for slightly faster."
            ))
            return

        # -- Long text warning (will be slow on CPU) --
        _TEXT_WARN_LIMIT = 2000
        warn_msg = None
        if len(text) > _TEXT_WARN_LIMIT:
            warn_msg = (f"Text is {len(text)} characters — synthesis may be slow on CPU. "
                        f"Consider splitting into chunks under {_TEXT_WARN_LIMIT} chars.")

        mode_vi = body.get("mode_vi", "standard")
        if mode_vi not in ("standard", "turbo"):
            mode_vi = "standard"

        try:
            from core.router import RouteConfig
            cfg = RouteConfig(lang=lang, speed_en=speed, speed_vi=speed, speed_ja=speed,
                              mode_vi=mode_vi)
            if voice:
                if voice.startswith(("am_", "af_", "bm_", "bf_")):
                    cfg.voice_en = voice
                elif voice.startswith("jvnv"):
                    cfg.voice_ja = voice
                else:
                    cfg.voice_vi = voice   # VI preset voice ID

            with _router_lock:
                router = _get_router()
                audio, sr = router.route(text, cfg)

            from core.audio_utils import normalize_audio
            audio = normalize_audio(audio, target_db=-20.0)

            if fmt == "MP3":
                import tempfile
                from core.audio_utils import export_mp3
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                    mp3_path = Path(tf.name)
                try:
                    export_mp3(audio, sr, mp3_path, bitrate="192k")
                    audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode("utf-8")
                finally:
                    mp3_path.unlink(missing_ok=True)
            else:
                audio_b64 = _audio_to_b64_wav(audio, sr)

            duration = len(audio) / sr
            response = {
                "audio_b64": audio_b64,
                "duration":  round(duration, 3),
                "format":    fmt,
                "message":   f"Synthesized {duration:.2f}s of audio.",
            }
            if warn_msg:
                response["warning"] = warn_msg
            _send_json(self, 200, _ok(response))

        except ValueError as exc:
            _send_json(self, 400, _err(str(exc)))
        except ImportError as exc:
            msg = str(exc)
            fix = (
                "pip install vieneu"
                " | Hướng dẫn: https://github.com/pnnbao97/VieNeu-TTS"
                if "vieneu" in msg.lower() or "VieNeu" in msg
                else "Chạy: python setup.py để cài engine còn thiếu."
            )
            _send_json(self, 503, _err(msg, fix=fix))
        except (PermissionError, OSError) as exc:
            _send_json(self, 500, _err(
                f"File system error during synthesis: {exc}",
                fix="Check that the output directory exists and is writable."
            ))
        except RuntimeError as exc:
            _send_json(self, 500, _err(
                f"Engine runtime error: {exc}",
                fix="The TTS model may not be downloaded yet. "
                    "Run python setup.py or check your internet connection."
            ))

    def _handle_batch(self, body: dict):
        """
        TASK-702: Accept batch job → return job_id immediately.
        Processing runs in a background thread; client polls /api/job/<id>.
        """
        files         = body.get("files", [])
        voice_config  = body.get("voice_config", {})
        fmt           = body.get("format", "WAV").upper()
        export_dir    = body.get("export_dir", str(_export_dir))

        if not files:
            _send_json(self, 400, _err("'files' list is required and must not be empty."))
            return
        if fmt not in ("WAV", "MP3"):
            _send_json(self, 400, _err(
                "'format' must be 'WAV' or 'MP3'.",
                fix="Set format to 'WAV' (recommended) or 'MP3'."
            ))
            return

        # Validate every file entry has at minimum a 'name' field
        for i, f in enumerate(files):
            if not isinstance(f, dict):
                _send_json(self, 400, _err(
                    f"files[{i}] must be an object with 'name' and 'content' fields."
                ))
                return
            if not f.get("name"):
                _send_json(self, 400, _err(
                    f"files[{i}].name is required.",
                    fix="Set 'name' to the original filename, e.g. 'chapter1.txt'."
                ))
                return

        # Validate export_dir is writable before starting the job
        try:
            from core.audio_utils import validate_output_dir
            validate_output_dir(export_dir)
        except (PermissionError, NotADirectoryError) as exc:
            _send_json(self, 400, _err(
                f"Export directory error: {exc}",
                fix=f"Ensure '{export_dir}' exists and is writable, "
                    f"or choose a different export_dir."
            ))
            return

        job_id = str(uuid.uuid4())
        _job_store.create(job_id)
        _job_store.update(job_id, total=len(files), status="running")

        # Launch background thread — processing is sequential inside it
        t = threading.Thread(
            target=_run_batch_job,
            args=(job_id, files, voice_config, fmt, export_dir),
            daemon=True,
        )
        t.start()

        _send_json(self, 202, _ok({"job_id": job_id, "total": len(files)}))


# ---------------------------------------------------------------------------
# Background batch job runner  (TASK-702)
# ---------------------------------------------------------------------------

def _run_batch_job(
    job_id: str,
    files: list[dict],
    voice_config: dict,
    fmt: str,
    export_dir: str,
) -> None:
    """Runs in a daemon thread. Updates _job_store as each file completes."""
    from batch.processor import BatchProcessor, BatchConfig

    cfg = BatchConfig(
        lang        = voice_config.get("lang",      "auto"),
        voice_en    = voice_config.get("voice_en",  "am_michael"),
        speed_en    = float(voice_config.get("speed_en",  1.3)),
        speed_vi    = float(voice_config.get("speed_vi",  1.0)),
        mode_vi     = voice_config.get("mode_vi",   "standard"),
        voice_ja    = voice_config.get("voice_ja",  "jvnv-M1-jp"),
        style_ja    = voice_config.get("style_ja",  "neutral"),
        speed_ja    = float(voice_config.get("speed_ja",  1.0)),
        format      = fmt,
        export_dir  = export_dir,
        normalize   = True,
        normalize_db= -20.0,
    )

    def on_progress(pct: float, name: str, done: int, total: int) -> None:
        _job_store.update(job_id,
            progress_pct = round(pct, 1),
            current_file = name,
            completed    = done,
        )

    try:
        proc   = BatchProcessor()
        result = proc.process(files, cfg, progress_callback=on_progress)

        serialized = []
        for r in result.results:
            serialized.append({
                "file_name":    r.file_name,
                "status":       r.status,
                "output_path":  str(r.output_path) if r.output_path else None,
                "duration":     round(r.duration, 3) if r.duration else None,
                "detected_lang":r.detected_lang,
                "error":        r.error,
            })

        _job_store.update(job_id,
            status       = "done",
            progress_pct = 100.0,
            results      = serialized,
            errors       = [r for r in serialized if r["status"] == "error"],
        )

    except Exception as exc:
        traceback.print_exc()
        _job_store.update(job_id,
            status = "error",
            errors = [{"file_name": "batch", "error": str(exc)}],
        )


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def start_server(
    port: int = DEFAULT_PORT,
    ui_dir: Path | None = None,
    export_dir: Path | None = None,
) -> HTTPServer:
    """
    Create and return an HTTPServer (not yet started — call .serve_forever()).

    Args:
        port:       TCP port to listen on. Default 8766.
        ui_dir:     Path to the ui/ directory containing index.html.
        export_dir: Default directory for batch audio exports.
    """
    global _ui_dir, _export_dir

    if ui_dir is not None:
        _ui_dir = Path(ui_dir)
    if export_dir is not None:
        _export_dir = Path(export_dir)

    server = HTTPServer(("127.0.0.1", port), TTSRequestHandler)
    return server
