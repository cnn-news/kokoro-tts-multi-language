import numpy as np
from pathlib import Path

SAMPLE_RATE = 44_100

# HuggingFace repo for JVNV pretrained models
_HF_REPO = "litagin/style_bert_vits2_jvnv"

# Available voices: model metadata + UI display info
VOICES: dict[str, dict] = {
    "jvnv-F1-jp": {
        "label": "⭐ JVNV F1 — Nữ, biểu cảm phong phú nhất",
        "group": "Nữ 👩 — Tiếng Nhật",
        "desc":  "Nữ · Tự nhiên, biểu cảm · Phong phú style nhất — recommended",
        "model_dir":  "jvnv-F1-jp",
        "model_file": "jvnv-F1-jp_e160_s14000.safetensors",
        "speaker":    "jvnv-F1-jp",
    },
    "jvnv-F2-jp": {
        "label": "JVNV F2 — Nữ, nhẹ nhàng trong trẻo",
        "group": "Nữ 👩 — Tiếng Nhật",
        "desc":  "Nữ · Giọng nhẹ nhàng, trong trẻo · Phù hợp anime/truyện",
        "model_dir":  "jvnv-F2-jp",
        "model_file": "jvnv-F2_e166_s20000.safetensors",
        "speaker":    "jvnv-F2-jp",
    },
    "jvnv-M1-jp": {
        "label": "⭐ JVNV M1 — Nam, trưởng thành chuyên nghiệp",
        "group": "Nam 👨 — Tiếng Nhật",
        "desc":  "Nam · Trưởng thành, chuyên nghiệp · Tốt cho narration/tin tức",
        "model_dir":  "jvnv-M1-jp",
        "model_file": "jvnv-M1-jp_e158_s14000.safetensors",
        "speaker":    "jvnv-M1-jp",
    },
    "jvnv-M2-jp": {
        "label": "JVNV M2 — Nam, trẻ trung thân thiện",
        "group": "Nam 👨 — Tiếng Nhật",
        "desc":  "Nam · Trẻ trung, thân thiện · Phù hợp hội thoại/casual",
        "model_dir":  "jvnv-M2-jp",
        "model_file": "jvnv-M2-jp_e159_s17000.safetensors",
        "speaker":    "jvnv-M2-jp",
    },
}

# Style names exposed to users → JVNV internal style names
STYLE_MAP: dict[str, str] = {
    "neutral":   "Neutral",
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "surprised": "Surprise",
    "disgust":   "Disgust",
    "fear":      "Fear",
}
STYLES = list(STYLE_MAP.keys())

# Model files cache dir inside the project
_MODEL_CACHE_DIR = Path(__file__).parent.parent / "models" / "ja"

# HuggingFace repo for the Japanese BERT model used by style-bert-vits2
_JP_BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
_BERT_LOADED = False   # module-level flag so we only load once per process


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class JapaneseTTSEngine:
    """
    Japanese TTS engine backed by Style-Bert-VITS2 with JVNV pretrained models.

    Models are downloaded from HuggingFace on first use per voice and cached
    locally under models/ja/.  Each voice is lazy-loaded and cached separately
    so switching voices does not reload the same model.
    """

    def __init__(self, device: str | None = None):
        self._device = device or _detect_device()
        self._models: dict[str, object] = {}   # voice_id → TTSModel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_id: str = "jvnv-M1-jp",
        style: str = "neutral",
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize Japanese text to audio (numpy float32, 44.1 kHz).

        Args:
            text:     Input text — Kanji, Hiragana, Katakana all supported.
            voice_id: One of the keys in VOICES.
            style:    Emotional style: neutral/happy/sad/angry/surprised/disgust/fear.
            speed:    Speed multiplier (0.5 – 2.0).  Mapped to VITS2 length scale.
        """
        self._validate(text, voice_id, style, speed)
        model = self._load_model(voice_id)
        internal_style = STYLE_MAP[style.lower()]

        # style-bert-vits2 length = 1/speed  (higher length → slower speech)
        length = 1.0 / speed

        sr, audio = model.infer(
            text=text,
            language="JP",
            style=internal_style,
            length=length,
            line_split=True,
            split_interval=0.3,
        )

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio

    def list_voices(self) -> dict[str, str]:
        """Return {voice_id: label} (backward compat)."""
        return {vid: info["label"] for vid, info in VOICES.items()}

    def list_voices_grouped(self) -> list[dict]:
        """Return [{id, label, group, desc}, ...] for UI optgroups."""
        return [
            {"id": vid, "label": info["label"],
             "group": info.get("group", "Tiếng Nhật 🇯🇵"),
             "desc":  info.get("desc", "")}
            for vid, info in VOICES.items()
        ]

    def list_styles(self) -> list[str]:
        """Return available emotional styles."""
        return list(STYLES)

    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, text: str, voice_id: str, style: str, speed: float) -> None:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if voice_id not in VOICES:
            raise ValueError(
                f"Unknown voice_id {voice_id!r}. "
                f"Available: {list(VOICES.keys())}"
            )
        if style.lower() not in STYLE_MAP:
            raise ValueError(
                f"Unknown style {style!r}. "
                f"Available: {STYLES}"
            )
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"speed must be in [0.5, 2.0], got {speed}")

    def _load_model(self, voice_id: str):
        """Lazy-download and load model; cache per voice_id."""
        if voice_id in self._models:
            return self._models[voice_id]

        try:
            from style_bert_vits2.tts_model import TTSModel
            from style_bert_vits2.constants import Languages
        except ImportError:
            raise ImportError(
                "style-bert-vits2 is not installed.\n"
                "Run: pip install style-bert-vits2"
            )

        info = VOICES[voice_id]
        model_dir_name = info["model_dir"]
        model_filename = info["model_file"]

        model_path, config_path, style_vec_path = self._ensure_model_files(
            model_dir_name, model_filename
        )

        self._ensure_bert_loaded()

        tts = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=self._device,
        )
        tts.load()
        self._models[voice_id] = tts
        return tts

    @staticmethod
    def _ensure_bert_loaded() -> None:
        """Download and cache the Japanese BERT model/tokenizer (once per process)."""
        global _BERT_LOADED
        if _BERT_LOADED:
            return

        import torch
        from style_bert_vits2.nlp import bert_models
        from style_bert_vits2.constants import Languages

        bert_model = bert_models.load_model(
            Languages.JP,
            pretrained_model_name_or_path=_JP_BERT_REPO,
        )
        # Ensure float32 — avoids FP16/float32 dtype mismatch during infer
        bert_model.to(torch.float32)

        bert_models.load_tokenizer(
            Languages.JP,
            pretrained_model_name_or_path=_JP_BERT_REPO,
        )
        _BERT_LOADED = True

    def _ensure_model_files(
        self, model_dir_name: str, model_filename: str
    ) -> tuple[Path, Path, Path]:
        """Download model files from HuggingFace if not already cached locally."""
        from huggingface_hub import hf_hub_download

        local_dir = _MODEL_CACHE_DIR / model_dir_name
        local_dir.mkdir(parents=True, exist_ok=True)

        def _dl(filename: str) -> Path:
            cached = hf_hub_download(
                repo_id=_HF_REPO,
                filename=f"{model_dir_name}/{filename}",
                local_dir=str(_MODEL_CACHE_DIR),
            )
            return Path(cached)

        model_path    = _dl(model_filename)
        config_path   = _dl("config.json")
        style_vec_path = _dl("style_vectors.npy")

        return model_path, config_path, style_vec_path
