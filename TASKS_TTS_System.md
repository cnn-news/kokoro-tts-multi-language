# 🎙️ TASKS — Hệ Thống TTS Đa Ngôn Ngữ (EN / VI / JA)

> Dùng với **Claude Code**. Mỗi task là một đơn vị công việc độc lập, thực hiện theo thứ tự từ trên xuống.
> Kiến trúc: Python backend + Web UI (localhost) + 3 engine TTS riêng biệt theo ngôn ngữ.

---

## 📐 KIẾN TRÚC TỔNG QUAN

```
tts_studio/
├── core/
│   ├── engine_en.py        # Kokoro TTS  — Tiếng Anh
│   ├── engine_vi.py        # VieNeu-TTS  — Tiếng Việt
│   ├── engine_ja.py        # Fish Audio S2 / Style-Bert-VITS2 — Tiếng Nhật
│   ├── router.py           # Tự động detect ngôn ngữ → chọn engine
│   └── audio_utils.py      # Normalize, export WAV/MP3
├── api/
│   └── server.py           # HTTP server (localhost)
├── ui/
│   └── index.html          # Web UI (dark theme, single-file)
├── batch/
│   └── processor.py        # Xử lý hàng loạt file .txt
├── config.py               # Cấu hình giọng, tốc độ, đường dẫn
├── requirements.txt
└── main.py                 # Entry point
```

**Engine theo ngôn ngữ:**
| Ngôn ngữ | Engine | Lý do |
|---|---|---|
| 🇺🇸 Tiếng Anh | **Kokoro TTS** (`kokoro`) | Chất lượng cao nhất open-source, nhẹ 82M params |
| 🇻🇳 Tiếng Việt | **VieNeu-TTS** (`vieneu`) | Chuyên biệt tiếng Việt, voice cloning, offline |
| 🇯🇵 Tiếng Nhật | **Style-Bert-VITS2** hoặc **Fish Speech** | Giọng Nhật tự nhiên, hỗ trợ nhiều kiểu nói |

---

## PHASE 0 — CHUẨN BỊ MÔI TRƯỜNG

### TASK-001 · Tạo cấu trúc thư mục dự án
```
Tạo toàn bộ cây thư mục như kiến trúc tổng quan ở trên.
Tạo các file __init__.py cần thiết.
Tạo file .gitignore chuẩn cho Python.
```

### TASK-002 · Tạo file requirements.txt
```
Liệt kê đầy đủ các thư viện cần thiết:

# Core TTS engines
kokoro>=0.9.0          # Engine tiếng Anh
vieneu>=1.0.0          # Engine tiếng Việt
fish-speech            # Engine tiếng Nhật (option A)
# hoặc style-bert-vits2  # Engine tiếng Nhật (option B)

# Audio processing
soundfile>=0.12.0
numpy>=1.24.0
pydub>=0.25.0          # Xuất MP3

# Language detection
langdetect>=1.0.9
lingua-language-detector>=2.0.0   # Chính xác hơn cho VI/JA

# Web server
# (dùng http.server built-in, không cần thêm)

# Utilities
pathlib
tqdm                   # Progress bar batch processing
```

### TASK-003 · Tạo script cài đặt tự động `setup.py`
```
Script kiểm tra và cài đặt:
- Kiểm tra Python >= 3.10
- Kiểm tra ffmpeg (cần cho MP3 export)
- pip install từng nhóm thư viện
- Download model weights nếu cần (VieNeu, Style-Bert-VITS2)
- In hướng dẫn nếu có lỗi
- Chạy: python setup.py
```

---

## PHASE 1 — ENGINE TIẾNG ANH

### TASK-101 · Xây dựng `core/engine_en.py`
```
Class: EnglishTTSEngine

Tích hợp Kokoro TTS với:
- Lazy loading pipeline (chỉ load khi cần, cache sau đó)
- Hỗ trợ lang_code 'a' (US) và 'b' (UK)
- Danh sách giọng Nam VI-BEST: am_michael, am_fenrir, am_puck, bm_george, bm_fable
- Danh sách giọng đầy đủ: tất cả am_*, af_*, bm_*, bf_*
- Method: synthesize(text, voice_id, speed) -> numpy array
- Method: list_voices() -> dict {label: voice_id}
- Xử lý text dài: tự chunk theo \n+ pattern
- Error handling rõ ràng nếu kokoro chưa cài
```

### TASK-102 · Unit test engine tiếng Anh
```
File: tests/test_engine_en.py
Test:
- Synthesize đoạn text ngắn (<50 từ) → trả về numpy array
- Synthesize đoạn text dài (>200 từ) → không lỗi
- List voices trả về dict không rỗng
- Tốc độ render < 10s trên CPU cho 100 từ
```

---

## PHASE 2 — ENGINE TIẾNG VIỆT

### TASK-201 · Xây dựng `core/engine_vi.py`
```
Class: VietnameseTTSEngine

Tích hợp VieNeu-TTS với:
- Lazy loading model (model nặng, chỉ load 1 lần)
- Hỗ trợ mode="standard" (chất lượng cao) và mode="turbo" (nhanh hơn)
- Voice cloning: nhận reference audio path (3-5 giây)
- Default voices built-in nếu không có reference
- Method: synthesize(text, speed, mode, ref_audio_path=None) -> numpy array
- Method: list_modes() -> ["standard", "turbo"]
- Xử lý code-switching Việt-Anh (ví dụ: "Hệ thống dùng machine learning")
- Error handling: thông báo rõ nếu vieneu chưa cài
```

### TASK-202 · Unit test engine tiếng Việt
```
File: tests/test_engine_vi.py
Test:
- Synthesize câu thuần tiếng Việt
- Synthesize câu có code-switching Việt-Anh
- Synthesize với turbo mode
- Synthesize có dấu thanh điệu phức tạp (sắc, huyền, hỏi, ngã, nặng)
```

---

## PHASE 3 — ENGINE TIẾNG NHẬT

### TASK-301 · Nghiên cứu và chọn engine Nhật phù hợp
```
Đánh giá 2 lựa chọn:

Option A — Style-Bert-VITS2:
  pip install style-bert-vits2
  Ưu: Nhiều kiểu giọng (tự nhiên, anime, phát thanh viên)
  Nhược: Cần download model riêng (~500MB)

Option B — Fish Speech (fish-speech):
  pip install fish-speech
  Ưu: 80+ ngôn ngữ, voice cloning, chất lượng SOTA
  Nhược: Model nặng hơn (~4B params), cần GPU tốt cho realtime

→ Mặc định dùng Style-Bert-VITS2 (nhẹ hơn, chạy được CPU)
→ Fallback sang Fish Speech nếu có GPU
```

### TASK-302 · Xây dựng `core/engine_ja.py`
```
Class: JapaneseTTSEngine

Tích hợp Style-Bert-VITS2 (primary) với:
- Auto-detect GPU/CPU, chọn device phù hợp
- Hỗ trợ các kiểu giọng: neutral, happy, sad, angry, surprised
- Các giọng Nam chuyên nghiệp: JVNv1, JVNv2 (model pretrained)
- Method: synthesize(text, voice_id, style, speed) -> numpy array
- Method: list_voices() -> dict
- Method: list_styles() -> list
- Xử lý ký tự Kanji/Hiragana/Katakana chuẩn
- Tích hợp text normalization cho tiếng Nhật (pyopenjtalk)
- Error handling: gợi ý cài đặt cụ thể nếu thiếu thư viện
```

### TASK-303 · Unit test engine tiếng Nhật
```
File: tests/test_engine_ja.py
Test:
- Synthesize câu Hiragana đơn giản
- Synthesize câu có Kanji
- Synthesize câu có Katakana (ngoại lai)
- Kiểm tra các style: neutral, happy
```

---

## PHASE 4 — LANGUAGE ROUTER

### TASK-401 · Xây dựng `core/router.py`
```
Class: LanguageRouter

Chức năng:
- Auto-detect ngôn ngữ từ text đầu vào
  → Dùng lingua-language-detector (chính xác hơn langdetect cho VI/JA)
- Mapping: ENGLISH → EnglishTTSEngine
            VIETNAMESE → VietnameseTTSEngine
            JAPANESE → JapaneseTTSEngine
            UNKNOWN → hỏi user hoặc dùng English làm mặc định

- Method: detect_language(text) -> "en" | "vi" | "ja" | "unknown"
- Method: route(text, config) -> audio numpy array
- Method: route_batch(file_list, config) -> list[audio]

Xử lý edge cases:
- Text quá ngắn (<5 ký tự) → không detect được, báo lỗi
- Mixed language → ưu tiên ngôn ngữ chiếm tỉ lệ cao nhất
- User override: cho phép chỉ định ngôn ngữ thủ công
```

### TASK-402 · Unit test router
```
File: tests/test_router.py
Test:
- "Hello world" → "en"
- "Xin chào" → "vi"
- "こんにちは" → "ja"
- "안녕하세요" → "unknown" (Korean, không hỗ trợ)
- Text có dấu tiếng Việt đầy đủ → "vi" (không nhầm sang "en")
```

---

## PHASE 5 — AUDIO UTILITIES

### TASK-501 · Xây dựng `core/audio_utils.py`
```
Functions:

normalize_audio(audio_np, target_db=-20.0) -> numpy array
  → Chuẩn hóa âm lượng theo loudness target

export_wav(audio_np, sample_rate, output_path) -> Path
  → Dùng soundfile

export_mp3(audio_np, sample_rate, output_path, bitrate="192k") -> Path
  → Dùng pydub + ffmpeg
  → Fallback WAV nếu ffmpeg không có, kèm warning

merge_audio_chunks(chunks: list[np.ndarray], silence_ms=300) -> numpy array
  → Ghép nhiều đoạn audio với khoảng lặng giữa các đoạn

get_duration(audio_np, sample_rate) -> float (seconds)

validate_output_dir(path: str) -> Path
  → Tạo thư mục nếu chưa có, raise nếu không có quyền ghi
```

---

## PHASE 6 — BATCH PROCESSOR

### TASK-601 · Xây dựng `batch/processor.py`
```
Class: BatchProcessor

Xử lý hàng loạt file .txt:
- Input: list of {name: str, content: str, lang_override: str|None}
- Config: voice, speed, format (WAV/MP3), export_dir
- Tự động detect ngôn ngữ cho từng file (hoặc dùng lang_override)
- Chạy tuần tự (sequential) để tránh OOM khi nhiều model load cùng lúc
- Progress callback: trả về % hoàn thành sau mỗi file
- Output: list of {file_name, output_path, duration, status, error}
- Tên file output = tên file input (giữ nguyên stem, đổi extension)

Method: process(files, config, progress_callback=None) -> BatchResult
Method: process_single(name, text, config) -> SingleResult
```

---

## PHASE 7 — HTTP SERVER & API

### TASK-701 · Xây dựng `api/server.py`
```
HTTP server đơn giản (http.server built-in):
Port mặc định: 8766

Endpoints:
GET  /              → Serve file ui/index.html
GET  /download?path=...  → Download file audio

POST /api/detect_lang    → {text} → {lang: "en"|"vi"|"ja"|"unknown"}
POST /api/voices         → {lang} → {voices: [{label, id}]}
POST /api/synthesize     → {text, lang, voice, speed, format} → {audio_b64, duration, message}
POST /api/batch          → {files:[{name,content,lang}], voice_config, format, export_dir} → SSE stream hoặc polling

Error format chuẩn: {status: "error", message: "..."}
Success format:     {status: "ok", ...data}
```

### TASK-702 · Streaming progress cho batch API
```
Endpoint POST /api/batch trả về JSON polling:
- Client gửi request → nhận job_id
- Client poll GET /api/job/{job_id} mỗi 1 giây
- Server trả về {status, progress_pct, current_file, completed, errors}
- Khi done: {status: "done", results: [...]}
Lý do: tránh timeout với batch lớn (>10 file)
```

---

## PHASE 8 — WEB UI

### TASK-801 · Xây dựng `ui/index.html` — Layout tổng thể
```
Single-file HTML (CSS + JS inline), dark theme giống Kokoro TTS Studio.

Layout:
- Header: tên app "TTS Studio Pro", version badge, status dot
- Tab nav: [🎙️ Thu Giọng Nhanh] [📁 Batch Files]
- Main: sidebar (360px) + content pane (flex-1)
- Footer: thông tin, shortcuts

CSS variables: giống bộ màu cũ (--bg, --accent, --surface...)
Font: Inter + JetBrains Mono
```

### TASK-802 · Tab 1 — Thu Giọng Nhanh (single text)
```
Sidebar:
  - Dropdown "Ngôn ngữ": [Tự động detect] [🇺🇸 Tiếng Anh] [🇻🇳 Tiếng Việt] [🇯🇵 Tiếng Nhật]
  - Dropdown "Giọng đọc": cập nhật động theo ngôn ngữ chọn
  - Slider "Tốc độ": 0.5x – 2.0x
  - (Nếu VI): dropdown "Mode": [Standard] [Turbo]
  - (Nếu JA): dropdown "Style": [Neutral] [Happy] [Sad] [Angry]
  - Button "🎙️ Tạo Audio"

Content pane:
  - Textarea nhập text (placeholder gợi ý ví dụ từng ngôn ngữ)
  - Badge hiển thị ngôn ngữ detect được (realtime khi typing, debounce 800ms)
  - Progress bar khi đang render
  - Audio player (<audio> controls)
  - Status message (duration, voice, speed)
```

### TASK-803 · Tab 2 — Batch Files
```
Sidebar:
  - Dropdown "Ngôn ngữ batch": [Tự động detect mỗi file] [EN] [VI] [JA]
  - Dropdown "Giọng đọc" (theo ngôn ngữ chọn)
  - Slider "Tốc độ"
  - Radio "Định dạng": WAV / MP3
  - Input "Thư mục export"
  - Button "🎙️ Tạo Audio Hàng Loạt"

Content pane:
  - Drop zone: kéo thả nhiều file .txt (multiple)
  - File chips: hiển thị tên file + icon ngôn ngữ detect được + nút xoá
  - Progress: thanh tiến trình tổng + tên file đang xử lý
  - Log box: kết quả từng file (✅ tên.wav  3.2s | ❌ lỗi...)
```

### TASK-804 · JavaScript — Language detection badge realtime
```
Khi user gõ vào textarea (Tab 1):
- Debounce 800ms sau khi dừng gõ
- Gọi POST /api/detect_lang với text hiện tại
- Hiển thị badge: 🇺🇸 English / 🇻🇳 Tiếng Việt / 🇯🇵 日本語 / ❓ Không rõ
- Badge xuất hiện mượt (fade-in animation)
- Nếu ngôn ngữ thay đổi → tự cập nhật dropdown giọng đọc
```

### TASK-805 · JavaScript — Batch file processing với polling
```
Khi nhấn "Tạo Audio Hàng Loạt":
1. POST /api/batch → nhận job_id
2. Bắt đầu interval poll GET /api/job/{job_id} mỗi 1000ms
3. Cập nhật progress bar + label "Đang xử lý file X/N: tên_file.txt"
4. Khi mỗi file done → thêm dòng vào log box (scroll auto)
5. Khi status === "done" → dừng poll, hiển thị tổng kết
6. Nếu lỗi → highlight đỏ dòng lỗi trong log
```

---

## PHASE 9 — TÍCH HỢP & CONFIG

### TASK-901 · Xây dựng `config.py`
```
Dataclass Config với các trường:

# Server
PORT = 8766

# Engine defaults
DEFAULT_LANG = "auto"  # "auto" | "en" | "vi" | "ja"

# English (Kokoro)
EN_DEFAULT_VOICE = "am_michael"
EN_DEFAULT_SPEED = 1.30
EN_SAMPLE_RATE   = 24_000

# Vietnamese (VieNeu)
VI_DEFAULT_MODE  = "standard"  # "standard" | "turbo"
VI_DEFAULT_SPEED = 1.0
VI_SAMPLE_RATE   = 24_000
VI_REF_AUDIO_DIR = "./voices/vi_reference/"  # thư mục chứa file tham chiếu

# Japanese (Style-Bert-VITS2)
JA_DEFAULT_VOICE = "jvnv-F1-jp"
JA_DEFAULT_STYLE = "neutral"
JA_DEFAULT_SPEED = 1.0
JA_SAMPLE_RATE   = 44_100

# Audio output
DEFAULT_FORMAT   = "WAV"  # "WAV" | "MP3"
MP3_BITRATE      = "192k"
DEFAULT_EXPORT   = "./output/"

Load từ: config.json (nếu tồn tại) → override defaults
```

### TASK-902 · Xây dựng `main.py` — Entry point
```
Thực hiện khi chạy: python main.py

1. Parse args: --port, --no-browser, --config
2. Chạy setup check:
   - Kiểm tra từng engine có import được không
   - Báo cáo engine nào OK / thiếu (nhưng không crash nếu 1 engine thiếu)
   - Ví dụ: "⚠ VieNeu chưa cài → tiếng Việt không khả dụng"
3. Khởi động HTTP server (thread daemon)
4. Mở trình duyệt (webbrowser.open)
5. Print banner thông tin đẹp (version, engines available, URL)
6. Giữ process với t.join(), Ctrl+C để thoát
```

---

## PHASE 10 — POLISH & DOCUMENTATION

### TASK-1001 · Xử lý lỗi toàn diện
```
Tất cả API endpoints cần handle:
- Engine chưa cài → trả về message hướng dẫn cài cụ thể
- Text rỗng → báo lỗi validation
- File export dir không ghi được → báo lỗi với path cụ thể
- Model chưa download → hướng dẫn download
- ffmpeg không có → tự động fallback WAV + warning

Tất cả lỗi dùng format chuẩn:
{
  "status": "error",
  "message": "Mô tả lỗi cho user",
  "fix": "Hướng dẫn khắc phục (tuỳ chọn)"
}
```

### TASK-1002 · Tạo file README.md
```
Nội dung:
- Giới thiệu hệ thống
- Bảng engine theo ngôn ngữ (Anh/Việt/Nhật)
- Yêu cầu hệ thống (Python, RAM, GPU optional)
- Hướng dẫn cài đặt từng bước
- Hướng dẫn sử dụng CLI và Web UI
- Danh sách giọng đọc đầy đủ
- Keyboard shortcuts
- FAQ + Troubleshooting thường gặp
- Changelog
```

### TASK-1003 · Tạo file VOICES.md
```
Liệt kê đầy đủ giọng đọc cho từng engine:

## 🇺🇸 Tiếng Anh — Kokoro TTS
(bảng: ID, giới tính, accent, mô tả, grade chất lượng)

## 🇻🇳 Tiếng Việt — VieNeu-TTS
(bảng: mode, mô tả, tốc độ inference)

## 🇯🇵 Tiếng Nhật — Style-Bert-VITS2
(bảng: ID, giới tính, style hỗ trợ, mô tả)
```

### TASK-1004 · Tối ưu performance
```
- Cache pipeline của mỗi engine sau lần load đầu tiên
- Lazy load: chỉ import engine khi có request thực sự
- Giới hạn text input: cảnh báo nếu >2000 ký tự (sẽ chậm)
- Batch: xử lý tuần tự để tránh OOM, không dùng multiprocessing
- Log thời gian render mỗi file vào console
```

---

## 📋 THỨ TỰ THỰC HIỆN KHUYẾN NGHỊ

```
Giai đoạn 1 (Core):   000 → 001 → 002 → 003
Giai đoạn 2 (Engine): 101-102 → 201-202 → 301-303
Giai đoạn 3 (Router): 401-402 → 501
Giai đoạn 4 (Batch):  601
Giai đoạn 5 (API):    701-702
Giai đoạn 6 (UI):     801 → 802 → 803 → 804 → 805
Giai đoạn 7 (Final):  901 → 902 → 1001 → 1002 → 1003 → 1004
```

---

## ⚠️ LƯU Ý QUAN TRỌNG CHO CLAUDE CODE

```
1. Mỗi TASK chạy độc lập — test xong mới sang task tiếp theo
2. Sau mỗi engine task → chạy unit test ngay để phát hiện lỗi sớm
3. Style-Bert-VITS2 cần download model riêng (~500MB) — cần hướng dẫn user
4. VieNeu-TTS model nặng (~1-2GB) — lazy load bắt buộc
5. Kokoro pipeline 'a' và 'b' là 2 object riêng — cache cả 2
6. Sample rate khác nhau: EN/VI=24000Hz, JA=44100Hz
   → audio_utils phải handle resample khi ghép audio mixed lang
7. Ưu tiên chạy được trên CPU trước, GPU là bonus
8. Không dùng Flask/FastAPI — dùng http.server built-in để zero dependency
```

---

*TTS Studio Pro — Multilingual EN/VI/JA · Phiên bản kế hoạch v1.0*
