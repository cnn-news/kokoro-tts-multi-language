import re
import numpy as np


# Full catalog with grouping and description — used by the UI dropdown
VOICE_CATALOG = [
    # ── ⭐ Đề xuất ────────────────────────────────────────────────────────
    {"id": "am_michael",  "label": "⭐ Michael",   "group": "⭐ Đề xuất — Tiếng Anh",
     "desc": "Nam · Mỹ 🇺🇸 · Trầm ấm, chuyên nghiệp — phổ biến nhất"},
    {"id": "am_fenrir",   "label": "⭐ Fenrir",    "group": "⭐ Đề xuất — Tiếng Anh",
     "desc": "Nam · Mỹ 🇺🇸 · Mạnh mẽ, dứt khoát — tốt nhất cho diễn đọc"},
    {"id": "bm_george",   "label": "⭐ George",    "group": "⭐ Đề xuất — Tiếng Anh",
     "desc": "Nam · Anh 🇬🇧 · Trang trọng, chuẩn British — tốt nhất giọng UK"},
    {"id": "af_heart",    "label": "⭐ Heart",     "group": "⭐ Đề xuất — Tiếng Anh",
     "desc": "Nữ · Mỹ 🇺🇸 · Ấm áp, tự nhiên — tốt nhất giọng nữ"},
    {"id": "af_bella",    "label": "⭐ Bella",     "group": "⭐ Đề xuất — Tiếng Anh",
     "desc": "Nữ · Mỹ 🇺🇸 · Ngọt ngào, dịu dàng — rất phổ biến"},
    # ── Nam — Mỹ 🇺🇸 ─────────────────────────────────────────────────────
    {"id": "am_michael",  "label": "Michael",      "group": "Nam — Mỹ 🇺🇸",
     "desc": "Trầm ấm, chuyên nghiệp — mặc định"},
    {"id": "am_fenrir",   "label": "Fenrir",       "group": "Nam — Mỹ 🇺🇸",
     "desc": "Mạnh mẽ, dứt khoát — diễn đọc xuất sắc"},
    {"id": "am_puck",     "label": "Puck",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Trẻ trung, linh hoạt — tốt cho nội dung casual"},
    {"id": "am_adam",     "label": "Adam",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Bình thường, tự nhiên — giọng trung tính"},
    {"id": "am_echo",     "label": "Echo",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Vang, rõ nét — tốt cho podcast"},
    {"id": "am_eric",     "label": "Eric",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Điềm tĩnh, đáng tin — thích hợp hướng dẫn"},
    {"id": "am_liam",     "label": "Liam",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Hiện đại, trẻ trung — phù hợp content trẻ"},
    {"id": "am_onyx",     "label": "Onyx",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Sâu, uy quyền — tốt cho quảng cáo"},
    {"id": "am_santa",    "label": "Santa",        "group": "Nam — Mỹ 🇺🇸",
     "desc": "Ấm áp, vui vẻ — phù hợp nội dung gia đình"},
    {"id": "am_zeus",     "label": "Zeus",         "group": "Nam — Mỹ 🇺🇸",
     "desc": "Hùng hồn, mạnh mẽ — tốt cho trailer/thể thao"},
    # ── Nữ — Mỹ 🇺🇸 ─────────────────────────────────────────────────────
    {"id": "af_heart",    "label": "Heart",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Ấm áp, tự nhiên nhất — recommended cho hầu hết nội dung"},
    {"id": "af_bella",    "label": "Bella",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Ngọt ngào, dịu dàng — tốt cho audiobook"},
    {"id": "af_nicole",   "label": "Nicole",       "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Nhỏ nhẹ, gợi cảm — phù hợp ASMR"},
    {"id": "af_sarah",    "label": "Sarah",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Thân thiện, tươi sáng — tốt cho hội thoại"},
    {"id": "af_sky",      "label": "Sky",          "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Nhẹ nhàng, trong sáng — giọng trẻ"},
    {"id": "af_nova",     "label": "Nova",         "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Sắc nét, hiện đại — phù hợp tech/business"},
    {"id": "af_river",    "label": "River",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Mượt mà, chảy trôi — tốt cho thiền/thư giãn"},
    {"id": "af_alloy",    "label": "Alloy",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Trung tính, phổ quát — dùng được mọi thể loại"},
    {"id": "af_jessica",  "label": "Jessica",      "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Biểu cảm, sôi nổi — phù hợp kể chuyện"},
    {"id": "af_kore",     "label": "Kore",         "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Tinh tế, thanh lịch — tốt cho thời trang/lifestyle"},
    {"id": "af_aoede",    "label": "Aoede",        "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Truyền cảm, nghệ thuật — tốt cho thơ/văn học"},
    {"id": "af_leda",     "label": "Leda",         "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Điềm tĩnh, thanh thản — phù hợp meditation"},
    {"id": "af_stella",   "label": "Stella",       "group": "Nữ — Mỹ 🇺🇸",
     "desc": "Tươi sáng, tích cực — tốt cho giáo dục"},
    # ── Nam — Anh 🇬🇧 ─────────────────────────────────────────────────────
    {"id": "bm_george",   "label": "George",       "group": "Nam — Anh 🇬🇧",
     "desc": "Trang trọng, chuẩn British — best UK male"},
    {"id": "bm_fable",    "label": "Fable",        "group": "Nam — Anh 🇬🇧",
     "desc": "Kể chuyện, ấm áp — tốt cho audiobook"},
    {"id": "bm_daniel",   "label": "Daniel",       "group": "Nam — Anh 🇬🇧",
     "desc": "Rõ ràng, đáng tin — phù hợp tin tức"},
    {"id": "bm_lewis",    "label": "Lewis",        "group": "Nam — Anh 🇬🇧",
     "desc": "Trẻ trung, thân thiện — tốt cho casual"},
    # ── Nữ — Anh 🇬🇧 ─────────────────────────────────────────────────────
    {"id": "bf_alice",    "label": "Alice",        "group": "Nữ — Anh 🇬🇧",
     "desc": "Thanh lịch, chuẩn British — best UK female"},
    {"id": "bf_emma",     "label": "Emma",         "group": "Nữ — Anh 🇬🇧",
     "desc": "Thân thiện, dễ chịu — tốt cho hội thoại"},
    {"id": "bf_isabella", "label": "Isabella",     "group": "Nữ — Anh 🇬🇧",
     "desc": "Ấm áp, tinh tế — phù hợp lifestyle"},
    {"id": "bf_lily",     "label": "Lily",         "group": "Nữ — Anh 🇬🇧",
     "desc": "Nhẹ nhàng, nữ tính — giọng trẻ UK"},
]

# Flat dict for backward-compatible list_voices() → {label: voice_id}
# Uses the "⭐ Đề xuất" entries first to avoid duplicates, then rest
_SEEN: set = set()
VOICES: dict[str, str] = {}
for _v in VOICE_CATALOG:
    if _v["id"] not in _SEEN:
        VOICES[_v["label"]] = _v["id"]
        _SEEN.add(_v["id"])
del _SEEN, _v

SAMPLE_RATE = 24_000

# voice_id prefix → KPipeline lang_code ('a'=US, 'b'=UK)
_LANG_CODE = {"a": "a", "b": "b"}


class EnglishTTSEngine:
    def __init__(self):
        # Lazy-loaded pipelines keyed by lang_code ('a' or 'b')
        self._pipelines: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, voice_id: str = "am_michael", speed: float = 1.3) -> np.ndarray:
        """Return synthesized audio as a numpy float32 array (24 kHz)."""
        if not text.strip():
            raise ValueError("text must not be empty")

        pipeline = self._get_pipeline(voice_id)
        chunks = []

        for result in pipeline(text, voice=voice_id, speed=speed, split_pattern=r"\n+"):
            if result.output is not None and result.output.audio is not None:
                raw = result.output.audio
                # torch tensor → numpy without triggering copy-keyword deprecation
                if hasattr(raw, "detach"):
                    audio = raw.detach().cpu().numpy().astype(np.float32)
                else:
                    audio = np.asarray(raw, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro returned no audio — check voice_id or input text")

        return np.concatenate(chunks)

    def list_voices(self) -> dict[str, str]:
        """Return {label: voice_id} for all available voices (backward compat)."""
        return dict(VOICES)

    def list_voices_grouped(self) -> list[dict]:
        """Return full catalog: [{id, label, group, desc}, ...] for UI optgroups."""
        return list(VOICE_CATALOG)

    def sample_rate(self) -> int:
        return SAMPLE_RATE

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_pipeline(self, voice_id: str):
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro is not installed.\n"
                "Run: pip install kokoro>=0.9.0"
            )

        prefix = voice_id[0] if voice_id else "a"
        lang_code = _LANG_CODE.get(prefix, "a")

        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)

        return self._pipelines[lang_code]
