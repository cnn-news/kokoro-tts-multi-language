"""
Unit tests for api/server.py.

A real HTTPServer is started on a free port for each test class; the same
server instance is reused within a class to avoid per-test startup overhead.
"""

import base64
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.server import start_server, JobStore, _ok, _err

# ---------------------------------------------------------------------------
# Shared server fixture
# ---------------------------------------------------------------------------

_SERVER     = None
_SERVER_URL = None
_TMPDIR     = None


def setUpModule():
    global _SERVER, _SERVER_URL, _TMPDIR
    import tempfile
    _TMPDIR  = tempfile.TemporaryDirectory()
    _SERVER  = start_server(port=18767, export_dir=Path(_TMPDIR.name))
    t = threading.Thread(target=_SERVER.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    _SERVER_URL = "http://127.0.0.1:18767"


def tearDownModule():
    if _SERVER:
        _SERVER.shutdown()
    if _TMPDIR:
        _TMPDIR.cleanup()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str):
    try:
        with urllib.request.urlopen(_SERVER_URL + path, timeout=10) as r:
            body = r.read()
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, {"_raw": body.decode(errors="replace")}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _post(path: str, data: dict, timeout: int = 60):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        _SERVER_URL + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _poll_job(job_id: str, max_wait: int = 60) -> dict:
    """Poll /api/job/<id> until status is done or error, or timeout."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(1)
        _, body = _get(f"/api/job/{job_id}")
        if body.get("status") in ("done", "error"):
            return body
    return {"status": "timeout"}


# ---------------------------------------------------------------------------
# JobStore unit tests (no HTTP)
# ---------------------------------------------------------------------------

class TestJobStore(unittest.TestCase):

    def setUp(self):
        self.store = JobStore()

    def test_create_returns_initial_state(self):
        state = self.store.create("j1")
        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["progress_pct"], 0.0)
        self.assertEqual(state["results"], [])

    def test_get_returns_none_for_unknown(self):
        self.assertIsNone(self.store.get("no-such-id"))

    def test_get_returns_copy(self):
        self.store.create("j2")
        s1 = self.store.get("j2")
        s1["status"] = "mutated"
        s2 = self.store.get("j2")
        self.assertNotEqual(s2["status"], "mutated")

    def test_update_modifies_state(self):
        self.store.create("j3")
        self.store.update("j3", status="running", progress_pct=50.0)
        state = self.store.get("j3")
        self.assertEqual(state["status"], "running")
        self.assertAlmostEqual(state["progress_pct"], 50.0)

    def test_update_unknown_id_is_noop(self):
        self.store.update("ghost", status="done")   # must not raise

    def test_append_result_ok(self):
        self.store.create("j4")
        self.store.append_result("j4", {"file_name": "a.txt", "status": "ok"})
        state = self.store.get("j4")
        self.assertEqual(len(state["results"]), 1)
        self.assertEqual(len(state["errors"]),  0)

    def test_append_result_error_goes_to_errors(self):
        self.store.create("j5")
        self.store.append_result("j5", {"file_name": "b.txt", "status": "error", "error": "oops"})
        state = self.store.get("j5")
        self.assertEqual(len(state["errors"]), 1)

    def test_thread_safety(self):
        """Concurrent updates must not corrupt state."""
        self.store.create("j6")
        def worker():
            for i in range(50):
                self.store.update("j6", progress_pct=float(i))
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for th in threads: th.start()
        for th in threads: th.join()
        state = self.store.get("j6")
        self.assertIn("progress_pct", state)


# ---------------------------------------------------------------------------
# Helper functions (no HTTP)
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_ok_includes_status(self):
        r = _ok({"foo": "bar"})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["foo"], "bar")

    def test_err_includes_status_and_message(self):
        r = _err("something broke")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["message"], "something broke")

    def test_err_with_fix(self):
        r = _err("missing dep", fix="pip install xyz")
        self.assertEqual(r["fix"], "pip install xyz")

    def test_err_without_fix_has_no_fix_key(self):
        r = _err("msg")
        self.assertNotIn("fix", r)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestGetRoot(unittest.TestCase):

    def test_returns_200_or_404(self):
        code, _ = _get("/")
        self.assertIn(code, (200, 404))

    def test_404_body_is_json_error(self):
        code, body = _get("/")
        if code == 404:
            self.assertEqual(body.get("status"), "error")

    def test_unknown_path_returns_404(self):
        code, body = _get("/no/such/path")
        self.assertEqual(code, 404)
        self.assertEqual(body.get("status"), "error")


# ---------------------------------------------------------------------------
# POST /api/detect_lang
# ---------------------------------------------------------------------------

class TestDetectLang(unittest.TestCase):

    def test_english_detected(self):
        code, body = _post("/api/detect_lang", {"text": "Hello, how are you today?"})
        self.assertEqual(code, 200)
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(body.get("lang"), "en")

    def test_japanese_detected(self):
        code, body = _post("/api/detect_lang", {"text": "こんにちは、元気ですか？"})
        self.assertEqual(code, 200)
        self.assertEqual(body.get("lang"), "ja")

    def test_vietnamese_with_tones_detected(self):
        code, body = _post("/api/detect_lang",
                           {"text": "Hệ thống phân tích dữ liệu lớn và hiện đại."})
        self.assertEqual(code, 200)
        self.assertEqual(body.get("lang"), "vi")

    def test_empty_text_returns_400(self):
        code, body = _post("/api/detect_lang", {"text": ""})
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_missing_text_field_returns_400(self):
        code, body = _post("/api/detect_lang", {})
        self.assertEqual(code, 400)

    def test_too_short_text_returns_400(self):
        code, body = _post("/api/detect_lang", {"text": "hi"})
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_unsupported_language_returns_unknown(self):
        code, body = _post("/api/detect_lang", {"text": "안녕하세요, 반갑습니다."})
        self.assertEqual(code, 200)
        self.assertEqual(body.get("lang"), "unknown")

    def test_response_has_lang_field(self):
        _, body = _post("/api/detect_lang", {"text": "Hello world today."})
        self.assertIn("lang", body)

    def test_lang_value_is_valid_code(self):
        _, body = _post("/api/detect_lang", {"text": "Hello world today."})
        self.assertIn(body.get("lang"), ("en", "vi", "ja", "unknown"))


# ---------------------------------------------------------------------------
# POST /api/voices
# ---------------------------------------------------------------------------

class TestVoices(unittest.TestCase):

    def test_english_voices_nonempty(self):
        code, body = _post("/api/voices", {"lang": "en"})
        self.assertEqual(code, 200)
        self.assertGreater(len(body.get("voices", [])), 5)

    def test_english_voice_has_label_and_id(self):
        _, body = _post("/api/voices", {"lang": "en"})
        for v in body.get("voices", []):
            self.assertIn("label", v)
            self.assertIn("id",    v)

    def test_japanese_voices_four(self):
        code, body = _post("/api/voices", {"lang": "ja"})
        self.assertEqual(code, 200)
        self.assertEqual(len(body.get("voices", [])), 4)

    def test_vietnamese_returns_modes(self):
        code, body = _post("/api/voices", {"lang": "vi"})
        self.assertEqual(code, 200)
        ids = [v["id"] for v in body.get("voices", [])]
        self.assertIn("standard", ids)
        self.assertIn("turbo",    ids)

    def test_unsupported_lang_returns_400(self):
        code, body = _post("/api/voices", {"lang": "zh"})
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_response_has_voices_key(self):
        _, body = _post("/api/voices", {"lang": "en"})
        self.assertIn("voices", body)

    def test_status_ok_on_success(self):
        _, body = _post("/api/voices", {"lang": "en"})
        self.assertEqual(body.get("status"), "ok")


# ---------------------------------------------------------------------------
# POST /api/synthesize
# ---------------------------------------------------------------------------

class TestSynthesize(unittest.TestCase):

    def test_english_wav_returns_200(self):
        code, body = _post("/api/synthesize",
                           {"text": "Hello world.", "lang": "en",
                            "speed": 1.3, "format": "wav"})
        self.assertEqual(code, 200)
        self.assertEqual(body.get("status"), "ok")

    def test_response_has_audio_b64(self):
        _, body = _post("/api/synthesize",
                        {"text": "Audio base64.", "lang": "en", "format": "wav"})
        self.assertIn("audio_b64", body)
        self.assertGreater(len(body["audio_b64"]), 100)

    def test_audio_b64_is_valid_base64(self):
        _, body = _post("/api/synthesize",
                        {"text": "Valid base64 check.", "lang": "en", "format": "wav"})
        try:
            decoded = base64.b64decode(body["audio_b64"])
            self.assertGreater(len(decoded), 0)
        except Exception as e:
            self.fail(f"audio_b64 is not valid base64: {e}")

    def test_wav_b64_starts_with_riff_header(self):
        _, body = _post("/api/synthesize",
                        {"text": "RIFF header check.", "lang": "en", "format": "wav"})
        decoded = base64.b64decode(body["audio_b64"])
        self.assertEqual(decoded[:4], b"RIFF")

    def test_response_has_duration(self):
        _, body = _post("/api/synthesize",
                        {"text": "Duration field test.", "lang": "en", "format": "wav"})
        self.assertIn("duration", body)
        self.assertGreater(body["duration"], 0.0)

    def test_japanese_synthesis(self):
        code, body = _post("/api/synthesize",
                           {"text": "テストです。", "lang": "ja",
                            "speed": 1.0, "format": "wav"})
        self.assertEqual(code, 200)
        self.assertGreater(len(body.get("audio_b64", "")), 100)

    def test_empty_text_returns_400(self):
        code, body = _post("/api/synthesize", {"text": "", "lang": "en"})
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_whitespace_text_returns_400(self):
        code, body = _post("/api/synthesize", {"text": "   ", "lang": "en"})
        self.assertEqual(code, 400)

    def test_invalid_format_returns_400(self):
        code, body = _post("/api/synthesize",
                           {"text": "Format test.", "lang": "en", "format": "flac"})
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_mp3_format_returns_audio(self):
        code, body = _post("/api/synthesize",
                           {"text": "MP3 format test.", "lang": "en", "format": "mp3"})
        self.assertEqual(code, 200)
        self.assertGreater(len(body.get("audio_b64", "")), 100)


# ---------------------------------------------------------------------------
# POST /api/batch  +  GET /api/job/<id>   (TASK-702)
# ---------------------------------------------------------------------------

class TestBatchAndPolling(unittest.TestCase):

    def _batch(self, files, **kwargs):
        data = {"files": files, "format": "WAV"}
        data.update(kwargs)
        return _post("/api/batch", data)

    def _en_files(self, *texts) -> list[dict]:
        return [{"name": f"f{i}.txt", "content": t, "lang_override": "en"}
                for i, t in enumerate(texts)]

    # --- POST /api/batch ---

    def test_returns_202_with_job_id(self):
        code, body = self._batch(self._en_files("Hello batch."))
        self.assertEqual(code, 202)
        self.assertEqual(body.get("status"), "ok")
        self.assertIn("job_id", body)

    def test_job_id_is_nonempty_string(self):
        _, body = self._batch(self._en_files("Job ID test."))
        self.assertIsInstance(body.get("job_id"), str)
        self.assertGreater(len(body["job_id"]), 0)

    def test_total_in_response(self):
        files = self._en_files("A.", "B.", "C.")
        _, body = self._batch(files)
        self.assertEqual(body.get("total"), 3)

    def test_empty_files_returns_400(self):
        code, body = self._batch([])
        self.assertEqual(code, 400)
        self.assertEqual(body.get("status"), "error")

    def test_invalid_format_returns_400(self):
        code, body = self._batch(self._en_files("Test."), format="OGG")
        self.assertEqual(code, 400)

    # --- GET /api/job/<id> ---

    def test_unknown_job_returns_404(self):
        code, body = _get("/api/job/no-such-job-id-xyz")
        self.assertEqual(code, 404)
        self.assertEqual(body.get("status"), "error")

    def test_job_initially_running_or_queued(self):
        _, body = self._batch(self._en_files("Quick job."))
        job_id = body.get("job_id")
        time.sleep(0.1)
        _, state = _get(f"/api/job/{job_id}")
        self.assertIn(state.get("status"), ("queued", "running", "done"))

    def test_job_completes_with_done_status(self):
        _, body = self._batch(self._en_files("Completion test."))
        state = _poll_job(body["job_id"])
        self.assertEqual(state.get("status"), "done")

    def test_completed_job_has_results(self):
        _, body = self._batch(self._en_files("Results test.", "Second sentence."))
        state = _poll_job(body["job_id"])
        self.assertEqual(len(state.get("results", [])), 2)

    def test_completed_results_have_ok_status(self):
        _, body = self._batch(self._en_files("Good text.", "More good text."))
        state = _poll_job(body["job_id"])
        for r in state.get("results", []):
            self.assertEqual(r.get("status"), "ok")

    def test_progress_pct_reaches_100(self):
        _, body = self._batch(self._en_files("Progress check."))
        state = _poll_job(body["job_id"])
        self.assertAlmostEqual(state.get("progress_pct", 0), 100.0)

    def test_empty_content_file_recorded_as_error(self):
        files = [
            {"name": "ok.txt",  "content": "Good text.",  "lang_override": "en"},
            {"name": "bad.txt", "content": "",             "lang_override": "en"},
        ]
        _, body = self._batch(files)
        state = _poll_job(body["job_id"])
        statuses = {r["file_name"]: r["status"] for r in state.get("results", [])}
        self.assertEqual(statuses.get("ok.txt"),  "ok")
        self.assertEqual(statuses.get("bad.txt"), "error")

    def test_error_in_one_file_does_not_fail_job(self):
        files = [
            {"name": "a.txt", "content": "First.",   "lang_override": "en"},
            {"name": "b.txt", "content": "",          "lang_override": "en"},
            {"name": "c.txt", "content": "Third.",   "lang_override": "en"},
        ]
        _, body = self._batch(files)
        state = _poll_job(body["job_id"])
        self.assertEqual(state.get("status"), "done",
                         "Batch job should reach 'done' even if some files error")

    def test_job_state_has_expected_keys(self):
        _, body = self._batch(self._en_files("Key check."))
        state = _poll_job(body["job_id"])
        for key in ("status", "progress_pct", "completed", "total", "results", "errors"):
            self.assertIn(key, state, f"Missing key: {key}")

    def test_multiple_jobs_independent(self):
        """Two separate batch jobs must not interfere with each other."""
        _, b1 = self._batch(self._en_files("Job one."))
        _, b2 = self._batch(self._en_files("Job two."))
        s1 = _poll_job(b1["job_id"])
        s2 = _poll_job(b2["job_id"])
        self.assertEqual(s1.get("status"), "done")
        self.assertEqual(s2.get("status"), "done")


if __name__ == "__main__":
    unittest.main(verbosity=2)
