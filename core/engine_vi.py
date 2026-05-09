"""
Vietnamese TTS engine — VieNeu-TTS v2
https://github.com/pnnbao97/VieNeu-TTS

Install (Windows CPU):
    pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
Install (Linux / macOS):
    pip install vieneu

API v2 thay đổi hoàn toàn so với v1:
    from vieneu import Vieneu
    tts = Vieneu()                         # lazy-load model
    audio = tts.infer(text="...", voice=v) # returns numpy float32, 24 kHz
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

SAMPLE_RATE = 24_000
MODES = ["standard", "turbo"]

# ── Danh sách giọng preset (7 giọng, 4 nam + 3 nữ) ───────────────────────
VOICE_CATALOG = [
    # Nam miền Bắc
    {"id": "Binh",  "label": "👨 Thanh Bình   — Nam, Bắc",  "group": "🇻🇳 Nam miền Bắc",  "desc": "Giọng nam ấm, chuẩn Bắc"},
    {"id": "Tuyen", "label": "👨 Phạm Tuyên  — Nam, Bắc",  "group": "🇻🇳 Nam miền Bắc",  "desc": "Giọng nam trầm, chuẩn Bắc"},
    # Nam miền Nam
    {"id": "Vinh",  "label": "👨 Xuân Vĩnh   — Nam, Nam",   "group": "🇻🇳 Nam miền Nam",   "desc": "Giọng nam miền Nam"},
    {"id": "Sơn",   "label": "👨 Thái Sơn    — Nam, Nam",   "group": "🇻🇳 Nam miền Nam",   "desc": "Giọng nam miền Nam"},
    # Nữ miền Bắc
    {"id": "Ly",    "label": "👩 Trúc Ly     — Nữ, Bắc ⭐", "group": "🇻🇳 Nữ miền Bắc",   "desc": "Giọng nữ mặc định, chuẩn Bắc"},
    {"id": "Ngoc",  "label": "👩 Bích Ngọc   — Nữ, Bắc",   "group": "🇻🇳 Nữ miền Bắc",   "desc": "Giọng nữ chuẩn Bắc"},
    # Nữ miền Nam
    {"id": "Doan",  "label": "👩 Thục Đoan   — Nữ, Nam",   "group": "🇻🇳 Nữ miền Nam",    "desc": "Giọng nữ miền Nam"},
]

# Set các voice ID hợp lệ để server detect nhanh
VI_VOICE_IDS: set[str] = {v["id"] for v in VOICE_CATALOG}

# Mode catalog — dùng cho dropdown chất lượng riêng
MODE_CATALOG = [
    {"id": "standard", "label": "⭐ Standard — Chất lượng cao", "desc": "Chất lượng tốt nhất"},
    {"id": "turbo",    "label": "⚡ Turbo — Nhanh 2×",          "desc": "Nhanh hơn, tiết kiệm CPU"},
]

INSTALL_GUIDE = (
    "VieNeu-TTS chưa được cài đặt. "
    "Chạy: pip install vieneu "
    "hoặc xem: https://github.com/pnnbao97/VieNeu-TTS"
)


# ─────────────────────────────────────────────────────────────────────────────

def _has_vieneu() -> bool:
    try:
        from vieneu import Vieneu  # noqa: F401
        return True
    except ImportError:
        return False


def _to_numpy(audio) -> np.ndarray:
    """Chuyển output của tts.infer() sang numpy float32."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if hasattr(audio, "numpy"):          # torch.Tensor
        return audio.numpy().astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Thay đổi tốc độ bằng time-stretching (librosa là dep của vieneu)."""
    if abs(speed - 1.0) < 0.02:
        return audio
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)
    except Exception:
        return audio   # nếu lỗi, trả về audio gốc


# ─────────────────────────────────────────────────────────────────────────────

class VietnameseTTSEngine:
    """
    Vietnamese TTS engine backed by VieNeu-TTS v2.
    Lazy-loads model on first synthesize() call.
    Supports voice cloning via a 3–5 second reference WAV.
    VieNeu v2 handles bilingual EN/VI natively — không cần code-switching thủ công.
    """

    def __init__(self) -> None:
        self._tts = None
        self._current_mode: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        mode: str = "standard",
        voice_id: str | None = None,
        ref_audio_path: str | None = None,
    ) -> np.ndarray:
        """
        Tổng hợp tiếng Việt (hoặc song ngữ EN/VI) sang numpy float32, 24 kHz.

        Args:
            text:           Văn bản đầu vào (tiếng Việt, có thể lẫn tiếng Anh).
            speed:          Tốc độ đọc (0.5–2.0).
            mode:           "standard" hoặc "turbo".
            voice_id:       ID giọng preset (Ly, Binh, Tuyen, Vinh, Doan, Sơn, Ngoc).
            ref_audio_path: WAV 3–5 giây để clone giọng — ưu tiên hơn voice_id.
        """
        self._validate_input(text, speed, mode, ref_audio_path)

        if not _has_vieneu():
            raise ImportError(INSTALL_GUIDE)

        tts = self._load_tts(mode)

        # Thứ tự ưu tiên: ref_audio > preset voice > mặc định
        voice = None
        if ref_audio_path is not None:
            voice = tts.encode_reference(ref_audio_path)
        elif voice_id is not None:
            voice = tts.get_preset_voice(voice_id)

        audio = _to_numpy(tts.infer(text=text, voice=voice))

        if abs(speed - 1.0) >= 0.02:
            audio = _apply_speed(audio, speed)

        return audio

    def list_modes(self) -> list[str]:
        return list(MODES)

    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_available(self) -> bool:
        return _has_vieneu()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_tts(self, mode: str):
        """Lazy-load Vieneu; reload nếu mode thay đổi."""
        if self._tts is None or self._current_mode != mode:
            from vieneu import Vieneu
            kwargs: dict = {}
            if mode == "turbo":
                kwargs["mode"] = "turbo"
            self._tts = Vieneu(**kwargs)
            self._current_mode = mode
        return self._tts

    def _validate_input(
        self,
        text: str,
        speed: float,
        mode: str,
        ref_audio_path: str | None,
    ) -> None:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"speed must be in [0.5, 2.0], got {speed}")
        if ref_audio_path is not None:
            p = Path(ref_audio_path)
            if not p.exists():
                raise FileNotFoundError(f"ref_audio_path not found: {ref_audio_path}")
            if p.suffix.lower() not in {".wav", ".mp3", ".flac"}:
                raise ValueError(
                    f"ref_audio_path must be a WAV/MP3/FLAC file: {ref_audio_path}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Kept for backward-compat with existing tests (VieNeu v2 handles this natively)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

_EN_TOKEN_RE = _re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
_VI_TONE_RE  = _re.compile(
    r"[àáảãạăắặẵẳặâấậẫẩèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẶẴẲẶÂẤẬẪẨÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐĐ]"
)
_VI_BARE_WORDS = {
    "an","ba","ban","bao","bay","bo","ca","cam","can","cho","chu","con","cua",
    "da","den","di","do","duc","em","gan","gap","gia","giua","go","goi","ha",
    "hai","han","hay","hoa","hoc","hoi","hon","hua","hung","huong","it","ke",
    "khi","khoa","khong","kia","kien","la","lai","lam","lan","len","lo","loi",
    "lop","lua","ma","me","moi","mot","muon","na","nam","nay","nha","nhieu",
    "nhu","no","noi","nua","oi","on","phan","qua","ra","rang","roi","san",
    "sau","se","so","sua","ta","tai","tan","tap","the","thi","thong","thu",
    "tien","tim","toan","toi","tren","trong","tu","tuan","van","vao","ve",
    "vi","viec","vien","vo","xin","xuat","y","yeu",
}


def _segment_for_codeswitching(text: str) -> list[dict]:
    """
    Tách văn bản thành các đoạn 'vi' / 'en'.
    Deprecated — VieNeu v2 xử lý song ngữ natively; giữ lại để test không vỡ.
    """
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format="text").split()
    except ImportError:
        tokens = text.split()

    segments: list[dict] = []
    for tok in tokens:
        is_latin  = bool(_EN_TOKEN_RE.fullmatch(tok))
        has_tone  = bool(_VI_TONE_RE.search(tok))
        is_bare   = tok.lower() in _VI_BARE_WORDS
        lang = "en" if (is_latin and not has_tone and not is_bare) else "vi"
        if segments and segments[-1]["lang"] == lang:
            segments[-1]["text"] += " " + tok
        else:
            segments.append({"lang": lang, "text": tok})
    return segments
