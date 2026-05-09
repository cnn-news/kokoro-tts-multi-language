# TTS Studio Pro — Hệ thống Text-to-Speech Đa Ngôn Ngữ

Tổng hợp giọng nói chất lượng cao cho **Tiếng Anh**, **Tiếng Việt** và **Tiếng Nhật** — chạy hoàn toàn offline trên máy cá nhân, giao diện Web UI tích hợp.

---

## Engines theo ngôn ngữ

| Ngôn ngữ | Engine | Model size | CPU? | Giọng |
|---|---|---|---|---|
| 🇺🇸 Tiếng Anh | **Kokoro TTS** | ~300 MB | ✅ | 31 giọng (US/UK, Nam/Nữ) |
| 🇻🇳 Tiếng Việt | **VieNeu-TTS** | ~1–2 GB | ✅ | Voice cloning, Standard/Turbo |
| 🇯🇵 Tiếng Nhật | **Style-Bert-VITS2** | ~170 MB/voice | ✅ | 4 giọng JVNV, 7 cảm xúc |

---

## Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| Python | 3.10+ | 3.12 |
| RAM | 4 GB | 8 GB+ |
| Disk | 3 GB (EN+JA) | 5 GB (EN+VI+JA) |
| GPU | Không bắt buộc | CUDA — nhanh hơn 3–5× |
| ffmpeg | Không bắt buộc | Cần thiết để xuất MP3 |
| OS | Windows / macOS / Linux | — |

---

## Cài đặt

### Bước 1 — Clone & tạo môi trường ảo

```bash
git clone <repo-url>
cd tts_studio
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Bước 2 — Chạy setup tự động

```bash
python setup.py
```

Script sẽ:
- Kiểm tra Python ≥ 3.10 và ffmpeg
- Cài tất cả thư viện Python cần thiết
- Tải model Style-Bert-VITS2 (JVNV) từ HuggingFace (~170 MB)
- Báo cáo engine nào sẵn sàng

### Bước 3 — (Tuỳ chọn) Cài VieNeu-TTS cho Tiếng Việt

VieNeu-TTS chưa có trên PyPI, cài thủ công:

```bash
pip install git+https://github.com/tts-vieneu/VieNeu-TTS.git
```

### Bước 4 — (Tuỳ chọn) Cài ffmpeg để xuất MP3

```bash
# Windows (winget):
winget install Gyan.FFmpeg

# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt install ffmpeg
```

---

## Khởi động

```bash
python main.py
```

Trình duyệt sẽ tự động mở `http://localhost:8766`.

### Tuỳ chọn CLI

```bash
python main.py --port 9000          # Đổi port
python main.py --no-browser         # Không mở trình duyệt
python main.py --config my.json     # Dùng file cấu hình tuỳ chỉnh
python main.py --help               # Xem tất cả tuỳ chọn
```

---

## Hướng dẫn sử dụng

### Tab 1 — Thu Giọng Nhanh

1. Chọn **Ngôn ngữ** (hoặc để "Tự động detect")
2. Chọn **Giọng đọc** từ danh sách
3. Điều chỉnh **Tốc độ** (0.5× – 2.0×)
4. Nhập văn bản vào ô lớn
5. Nhấn **🎙️ Tạo Audio** hoặc `Ctrl+Enter`
6. Nghe ngay trong trình phát âm thanh tích hợp

### Tab 2 — Batch Files

1. Kéo & thả nhiều file `.txt` vào vùng drop
2. Hệ thống tự detect ngôn ngữ từng file
3. Chọn giọng đọc, tốc độ, định dạng (WAV/MP3)
4. Nhập thư mục xuất file audio
5. Nhấn **🎙️ Tạo Audio Hàng Loạt**
6. Theo dõi tiến trình theo thời gian thực

### Cấu hình tuỳ chỉnh (`config.json`)

Tạo file `config.json` cạnh `main.py` để override defaults:

```json
{
  "PORT": 9000,
  "EN_DEFAULT_VOICE": "am_fenrir",
  "EN_DEFAULT_SPEED": 1.5,
  "JA_DEFAULT_STYLE": "happy",
  "DEFAULT_FORMAT": "MP3",
  "DEFAULT_EXPORT": "D:/my_audio_output"
}
```

---

## API HTTP

Server chạy tại `http://localhost:8766` với các endpoint:

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/download?path=...` | Tải file audio |
| GET | `/api/job/<id>` | Poll trạng thái batch job |
| POST | `/api/detect_lang` | `{text}` → `{lang}` |
| POST | `/api/voices` | `{lang}` → `{voices}` |
| POST | `/api/synthesize` | Tổng hợp giọng đọc |
| POST | `/api/batch` | Bắt đầu batch job |

**Ví dụ — Synthesize:**
```bash
curl -X POST http://localhost:8766/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","lang":"en","speed":1.3,"format":"wav"}'
```

---

## Keyboard Shortcuts

| Phím tắt | Hành động |
|---|---|
| `Ctrl+Enter` | Tạo audio (Tab 1) |
| `Ctrl+L` | Xoá text (Tab 1) |

---

## Danh sách giọng đọc

Xem chi tiết tại [VOICES.md](VOICES.md).

---

## FAQ & Troubleshooting

**Q: Lần đầu chạy rất chậm?**  
A: Kokoro và Style-Bert-VITS2 tải model weights lần đầu (~vài giây). Các lần sau pipeline được cache, render chỉ mất 0.5–3s.

**Q: Lỗi "Engine not available"?**  
A: Chạy `python setup.py` để kiểm tra và cài các thư viện còn thiếu.

**Q: Xuất MP3 không được, chỉ có WAV?**  
A: ffmpeg chưa được cài. Xem [Bước 4](#bước-4--tuỳ-chọn-cài-ffmpeg-để-xuất-mp3) ở trên.

**Q: Tiếng Việt chưa khả dụng?**  
A: VieNeu-TTS cần cài thủ công (chưa có trên PyPI). Xem [Bước 3](#bước-3--tuỳ-chọn-cài-vieneu-tts-cho-tiếng-việt).

**Q: Port 8766 đã bị chiếm?**  
A: Chạy `python main.py --port 8767` để dùng port khác.

**Q: Text dài (>2000 ký tự) render chậm?**  
A: CPU inference chậm hơn GPU. Nên chia text thành đoạn ngắn hơn, hoặc dùng GPU nếu có.

**Q: Style-Bert-VITS2 báo lỗi model chưa download?**  
A: Chạy `python setup.py` lại — script sẽ tải model từ HuggingFace. Cần kết nối internet lần đầu.

---

## Changelog

### v1.0.0
- 🇺🇸 Engine Tiếng Anh: Kokoro TTS — 31 giọng US/UK
- 🇻🇳 Engine Tiếng Việt: VieNeu-TTS — voice cloning, Standard/Turbo mode
- 🇯🇵 Engine Tiếng Nhật: Style-Bert-VITS2 — 4 giọng JVNV, 7 cảm xúc
- Auto-detect ngôn ngữ với lingua-language-detector
- Web UI dark theme, Tab 1 (single) + Tab 2 (batch)
- Batch processing với realtime polling
- Export WAV/MP3, normalize audio
- `python main.py` entry point với banner và engine check
