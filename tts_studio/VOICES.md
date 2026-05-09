# VOICES — Danh sách giọng đọc đầy đủ

---

## 🇺🇸 Tiếng Anh — Kokoro TTS

Kokoro TTS 82M params, chạy CPU. Sample rate: **24 000 Hz**.

### ⭐ Recommended (chất lượng cao nhất)

| Voice ID | Tên | Giới tính | Accent | Chất lượng |
|---|---|---|---|---|
| `am_michael` | Michael | Nam | US 🇺🇸 | ⭐⭐⭐⭐⭐ |
| `am_fenrir` | Fenrir | Nam | US 🇺🇸 | ⭐⭐⭐⭐⭐ |
| `am_puck` | Puck | Nam | US 🇺🇸 | ⭐⭐⭐⭐ |
| `bm_george` | George | Nam | UK 🇬🇧 | ⭐⭐⭐⭐⭐ |
| `bm_fable` | Fable | Nam | UK 🇬🇧 | ⭐⭐⭐⭐ |
| `af_heart` | Heart | Nữ | US 🇺🇸 | ⭐⭐⭐⭐⭐ |

### Nam — US English 🇺🇸

| Voice ID | Tên | Mô tả |
|---|---|---|
| `am_michael` | Michael | Trầm ấm, chuyên nghiệp — mặc định |
| `am_fenrir` | Fenrir | Mạnh mẽ, rõ ràng |
| `am_puck` | Puck | Trẻ trung, linh hoạt |
| `am_adam` | Adam | Bình thường, tự nhiên |
| `am_echo` | Echo | Giọng vang, rõ nét |
| `am_eric` | Eric | Điềm tĩnh, tin tưởng |
| `am_liam` | Liam | Trẻ, hiện đại |
| `am_onyx` | Onyx | Sâu, uy quyền |
| `am_santa` | Santa | Ấm áp, vui vẻ |
| `am_zeus` | Zeus | Hùng hồn, mạnh mẽ |

### Nữ — US English 🇺🇸

| Voice ID | Tên | Mô tả |
|---|---|---|
| `af_heart` | Heart | Ấm áp, dễ nghe — recommended |
| `af_bella` | Bella | Ngọt ngào, dịu dàng |
| `af_nicole` | Nicole | Chuyên nghiệp, rõ ràng |
| `af_sarah` | Sarah | Tự nhiên, thân thiện |
| `af_sky` | Sky | Sáng sủa, tươi trẻ |
| `af_nova` | Nova | Hiện đại, sắc nét |
| `af_river` | River | Mềm mại, chảy trôi |
| `af_alloy` | Alloy | Trung tính, phổ quát |
| `af_jessica` | Jessica | Năng động, biểu cảm |
| `af_kore` | Kore | Tinh tế, thanh lịch |
| `af_aoede` | Aoede | Truyền cảm, nghệ thuật |
| `af_leda` | Leda | Dịu dàng, điềm tĩnh |
| `af_stella` | Stella | Tươi sáng, tích cực |

### Nam — UK English 🇬🇧

| Voice ID | Tên | Mô tả |
|---|---|---|
| `bm_george` | George | British chuẩn, chuyên nghiệp |
| `bm_fable` | Fable | Truyện kể, ấm áp |
| `bm_daniel` | Daniel | Rõ ràng, đáng tin |
| `bm_lewis` | Lewis | Trẻ trung, hiện đại |

### Nữ — UK English 🇬🇧

| Voice ID | Tên | Mô tả |
|---|---|---|
| `bf_alice` | Alice | Thanh lịch, chuẩn British |
| `bf_emma` | Emma | Thân thiện, dễ chịu |
| `bf_isabella` | Isabella | Ấm áp, tinh tế |
| `bf_lily` | Lily | Nhẹ nhàng, nữ tính |

### Cách dùng

```python
from core.engine_en import EnglishTTSEngine
engine = EnglishTTSEngine()
audio = engine.synthesize("Hello world.", voice_id="am_michael", speed=1.3)
```

---

## 🇻🇳 Tiếng Việt — VieNeu-TTS

VieNeu-TTS, offline, hỗ trợ voice cloning. Sample rate: **24 000 Hz**.

> **Lưu ý:** VieNeu-TTS cần cài thủ công — xem [README.md](README.md#bước-3--tuỳ-chọn-cài-vieneu-tts-cho-tiếng-việt).

### Modes

| Mode ID | Tên | Mô tả | Tốc độ inference |
|---|---|---|---|
| `standard` | Standard | Chất lượng cao nhất, tự nhiên | ~2–4s/câu (CPU) |
| `turbo` | Turbo | Nhanh hơn 2×, chất lượng tốt | ~1–2s/câu (CPU) |

### Voice Cloning

Cung cấp file audio mẫu (WAV, 3–5 giây, giọng rõ, không tạp âm):

```python
from core.engine_vi import VietnameseTTSEngine
engine = VietnameseTTSEngine()
audio = engine.synthesize(
    "Xin chào, đây là giọng đọc của tôi.",
    mode="standard",
    ref_audio_path="./voices/vi_reference/my_voice.wav"
)
```

### Code-switching

Engine tự xử lý văn bản có cả tiếng Việt lẫn tiếng Anh:

```
"Hệ thống dùng machine learning để phân tích dữ liệu."
```

---

## 🇯🇵 Tiếng Nhật — Style-Bert-VITS2 (JVNV)

JVNV pretrained models từ `litagin/style_bert_vits2_jvnv`. Sample rate: **44 100 Hz**.  
Tự động download từ HuggingFace lần đầu sử dụng (~170 MB/model).

### Voices

| Voice ID | Tên | Giới tính | Mô tả |
|---|---|---|---|
| `jvnv-F1-jp` | JVNV Female 1 | Nữ | Giọng nữ tự nhiên — **mặc định** |
| `jvnv-F2-jp` | JVNV Female 2 | Nữ | Giọng nữ nhẹ nhàng |
| `jvnv-M1-jp` | JVNV Male 1 | Nam | Giọng nam trưởng thành |
| `jvnv-M2-jp` | JVNV Male 2 | Nam | Giọng nam trẻ trung |

### Styles (cảm xúc)

| Style | Tên tiếng Việt | Dùng khi |
|---|---|---|
| `neutral` | Bình thường | Nội dung phổ thông — mặc định |
| `happy` | Vui vẻ | Tin vui, lời chào, quảng cáo |
| `sad` | Buồn | Nội dung cảm xúc, phim |
| `angry` | Tức giận | Nhân vật phản diện, drama |
| `surprised` | Ngạc nhiên | Thông báo bất ngờ |
| `disgust` | Ghê tởm | Nhân vật phản cảm |
| `fear` | Sợ hãi | Horror, hồi hộp |

### Cách dùng

```python
from core.engine_ja import JapaneseTTSEngine
engine = JapaneseTTSEngine()

# Kanji + Hiragana + Katakana đều được hỗ trợ
audio = engine.synthesize(
    "東京のコンビニでスマートフォンを買いました。",
    voice_id="jvnv-M1-jp",
    style="neutral",
    speed=1.0
)
```

### Thư mục cache model

Model được lưu tại `models/ja/<voice_id>/` trong thư mục dự án.

---

## Tóm tắt nhanh

| Ngôn ngữ | Sample Rate | Giọng/Modes | Voice Cloning | GPU Optional |
|---|---|---|---|---|
| 🇺🇸 Tiếng Anh | 24 000 Hz | 31 giọng | ✗ | ✅ |
| 🇻🇳 Tiếng Việt | 24 000 Hz | 2 modes | ✅ (3–5s WAV) | ✅ |
| 🇯🇵 Tiếng Nhật | 44 100 Hz | 4 giọng × 7 style | ✗ | ✅ |
