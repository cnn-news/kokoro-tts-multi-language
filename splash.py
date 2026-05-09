"""
Màn hình khởi động — TTS Studio Pro
Kiểm tra và tự động cài đặt thư viện trước khi vào ứng dụng chính.

Sử dụng:
    from splash import SplashScreen
    SplashScreen().run()   # trả về True nếu hoàn tất
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

# ─────────────────────────────────────────────────────────────────────────────
# Danh sách gói cần kiểm tra / cài đặt
# (pip_spec, import_name, tên hiển thị)
# ─────────────────────────────────────────────────────────────────────────────
_PACKAGES: list[tuple[str, str, str]] = [
    ("numpy>=1.24.0",                    "numpy",            "NumPy"),
    ("soundfile>=0.12.0",                "soundfile",        "SoundFile"),
    ("pydub>=0.25.0",                    "pydub",            "PyDub"),
    ("langdetect>=1.0.9",                "langdetect",       "LangDetect"),
    ("lingua-language-detector>=2.0.0",  "lingua",           "Lingua Detector"),
    ("kokoro>=0.9.0",                    "kokoro",           "Kokoro TTS  EN"),
    ("style-bert-vits2",                 "style_bert_vits2", "Style-Bert-VITS2  JA"),
    ("pyopenjtalk",                      "pyopenjtalk",      "pyopenjtalk  JA"),
    ("tqdm>=4.65.0",                     "tqdm",             "TQDM"),
]

# VieNeu có trên PyPI — pip install vieneu
# Windows cần extra-index-url để tránh lỗi build llama-cpp
_VIENEU_PIP = (
    "vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/"
    if sys.platform == "win32"
    else "vieneu"
)
_VIENEU = (
    _VIENEU_PIP,
    "vieneu",
    "VieNeu TTS  VI",
)

# ─────────────────────────────────────────────────────────────────────────────
# Bảng màu dark-theme (khớp với giao diện web)
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "bg":     "#0d1117",
    "bg2":    "#161b22",
    "bg3":    "#21262d",
    "border": "#30363d",
    "fg":     "#e6edf3",
    "dim":    "#8b949e",
    "accent": "#58a6ff",
    "green":  "#3fb950",
    "yellow": "#d29922",
    "red":    "#f85149",
}

_W, _H = 580, 500   # kích thước cửa sổ splash


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — chạy trong worker thread (không cần lock, subprocess an toàn)
# ─────────────────────────────────────────────────────────────────────────────

def _is_importable(module: str) -> bool:
    """Kiểm tra xem module có thể import được không (dùng subprocess riêng)."""
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
    ).returncode == 0


def _pip_install(spec: str) -> bool:
    """Cài đặt gói qua pip, trả về True nếu thành công.
    spec có thể chứa extra args, vd: 'vieneu --extra-index-url https://...'
    """
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", *shlex.split(spec),
         "--quiet", "--disable-pip-version-check"],
        capture_output=True, text=True,
    ).returncode == 0


def _pip_install_verbose(spec: str) -> tuple[bool, str]:
    """Cài đặt gói qua pip, trả về (thành_công, dòng_lỗi_cuối)."""
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", *shlex.split(spec),
         "--disable-pip-version-check"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return True, ""
    # Lấy dòng lỗi có nghĩa nhất từ stderr/stdout
    stderr = (r.stderr or r.stdout or "").strip()
    last_line = next(
        (ln.strip() for ln in reversed(stderr.splitlines()) if ln.strip()),
        "pip thất bại (không rõ lý do)"
    )
    return False, last_line[:120]


# ─────────────────────────────────────────────────────────────────────────────
# Màn hình splash
# ─────────────────────────────────────────────────────────────────────────────

class SplashScreen:
    """Cửa sổ khởi động kiểm tra / cài đặt thư viện."""

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()                   # ẩn trước khi dựng xong
        self._root.title("TTS Studio Pro")
        self._root.overrideredirect(True)       # bỏ thanh tiêu đề OS
        self._root.configure(bg=_C["border"])   # border 1px từ màu nền
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)

        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.geometry(
            f"{_W}x{_H}+{(sw - _W) // 2}+{(sh - _H) // 2}"
        )

        self._drag_x = self._drag_y = 0
        self.success = False

        self._build_ui()

        # Cho phép kéo cửa sổ bằng chuột
        for widget in (self._root, self._header):
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>",     self._on_drag_move)

        self._root.deiconify()   # hiện cửa sổ

    # ─── Xây giao diện ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Khung trong (cách border 1px)
        inner = tk.Frame(self._root, bg=_C["bg"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        # ── Header ────────────────────────────────────────────────────────
        self._header = tk.Frame(inner, bg=_C["bg2"])
        self._header.pack(fill="x")

        # Nút đóng (×) góc phải
        close_lbl = tk.Label(
            self._header, text="  ×  ",
            font=("Segoe UI", 13), bg=_C["bg2"], fg=_C["dim"], cursor="hand2",
        )
        close_lbl.pack(side="right", padx=4, pady=4)
        close_lbl.bind("<Button-1>", lambda _: self._root.destroy())
        close_lbl.bind("<Enter>",    lambda _: close_lbl.config(fg=_C["red"]))
        close_lbl.bind("<Leave>",    lambda _: close_lbl.config(fg=_C["dim"]))

        # Logo + tiêu đề
        logo = tk.Frame(self._header, bg=_C["bg2"])
        logo.pack(pady=(18, 14))
        tk.Label(
            logo, text="◈  TTS Studio Pro",
            font=("Segoe UI", 20, "bold"), bg=_C["bg2"], fg=_C["fg"],
        ).pack()
        tk.Label(
            logo, text="Multilingual Text-to-Speech  ·  EN / VI / JA",
            font=("Segoe UI", 9), bg=_C["bg2"], fg=_C["dim"],
        ).pack(pady=(2, 0))

        # Đường kẻ accent
        tk.Frame(inner, bg=_C["accent"], height=2).pack(fill="x")

        # ── Trạng thái + thanh tiến trình ─────────────────────────────────
        mid = tk.Frame(inner, bg=_C["bg"])
        mid.pack(fill="x", padx=24, pady=(14, 6))

        self._status_var = tk.StringVar(value="Đang khởi động…")
        tk.Label(
            mid, textvariable=self._status_var,
            font=("Segoe UI", 10), bg=_C["bg"], fg=_C["accent"], anchor="w",
        ).pack(fill="x")

        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure(
            "Splash.Horizontal.TProgressbar",
            troughcolor=_C["bg3"], background=_C["accent"],
            bordercolor=_C["border"],
            lightcolor=_C["accent"], darkcolor=_C["accent"],
        )
        self._progress = ttk.Progressbar(
            mid, style="Splash.Horizontal.TProgressbar",
            length=_W - 48, mode="determinate", maximum=100,
        )
        self._progress.pack(fill="x", pady=(6, 0))

        # ── Hộp log ───────────────────────────────────────────────────────
        log_frame = tk.Frame(
            inner, bg=_C["bg2"],
            highlightbackground=_C["border"], highlightthickness=1,
        )
        log_frame.pack(fill="both", expand=True, padx=24, pady=10)

        self._log = tk.Text(
            log_frame, bg=_C["bg2"], fg=_C["fg"],
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word", cursor="arrow",
        )
        vsb = tk.Scrollbar(
            log_frame, command=self._log.yview,
            bg=_C["bg3"], troughcolor=_C["bg2"], relief="flat", width=8,
        )
        self._log.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True, padx=6, pady=6)

        # Tags màu sắc
        self._log.tag_configure("ok",      foreground=_C["green"])
        self._log.tag_configure("warn",    foreground=_C["yellow"])
        self._log.tag_configure("err",     foreground=_C["red"])
        self._log.tag_configure("info",    foreground=_C["accent"])
        self._log.tag_configure("dim",     foreground=_C["dim"])
        self._log.tag_configure("section", foreground=_C["fg"],
                                 font=("Consolas", 9, "bold"))

        # ── Footer ────────────────────────────────────────────────────────
        footer = tk.Frame(inner, bg=_C["bg3"])
        footer.pack(fill="x", side="bottom")
        tk.Label(
            footer,
            text=f"v1.0.0  ·  Python {sys.version.split()[0]}",
            font=("Segoe UI", 8), bg=_C["bg3"], fg=_C["dim"],
        ).pack(side="left", padx=12, pady=5)
        tk.Label(
            footer, text="Kéo để di chuyển  ·  × để đóng",
            font=("Segoe UI", 8), bg=_C["bg3"], fg=_C["dim"],
        ).pack(side="right", padx=12, pady=5)

    # ─── Kéo cửa sổ ───────────────────────────────────────────────────────

    def _on_drag_start(self, e: tk.Event) -> None:
        self._drag_x, self._drag_y = e.x_root, e.y_root

    def _on_drag_move(self, e: tk.Event) -> None:
        x = self._root.winfo_x() + (e.x_root - self._drag_x)
        y = self._root.winfo_y() + (e.y_root - self._drag_y)
        self._root.geometry(f"+{x}+{y}")
        self._drag_x, self._drag_y = e.x_root, e.y_root

    # ─── Ghi log (an toàn khi gọi từ main thread qua after()) ─────────────

    def _log_append(self, prefix: str, ptag: str,
                    text: str, ttag: str = "") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", f"  {prefix}  ", ptag)
        self._log.insert("end", f"{text}\n", ttag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_section(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", f"\n  {text}\n", "section")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_progress(self, value: int) -> None:
        self._progress["value"] = value

    # ─── Gọi an toàn từ worker thread ─────────────────────────────────────

    def _ui(self, fn, *args) -> None:
        """Lên lịch gọi hàm trên main thread (Tkinter không thread-safe)."""
        try:
            self._root.after(0, fn, *args)
        except tk.TclError:
            pass   # cửa sổ đã bị đóng

    # ─── Worker thread — chạy nền, cập nhật UI qua _ui() ──────────────────

    def _worker(self) -> None:
        total = len(_PACKAGES) + 1   # +1 cho vieneu

        # ── Kiểm tra ffmpeg ────────────────────────────────────────────────
        self._ui(self._log_section, "[ Kiểm tra hệ thống ]")
        self._ui(self._status_var.set, "Kiểm tra ffmpeg…")

        if shutil.which("ffmpeg"):
            self._ui(self._log_append, "✓", "ok",
                     "ffmpeg                         — tìm thấy trên PATH")
        else:
            self._ui(self._log_append, "⚠", "warn",
                     "ffmpeg chưa cài — xuất MP3 sẽ bị tắt")
            self._ui(self._log_append, "→", "info",
                     "winget install Gyan.FFmpeg", "dim")

        # ── Kiểm tra / cài đặt Python packages ────────────────────────────
        self._ui(self._log_section, "[ Thư viện Python ]")

        for idx, (pip_spec, import_name, display) in enumerate(_PACKAGES, start=1):
            pct = int(idx / total * 90)

            self._ui(self._status_var.set, f"Kiểm tra {display}…")
            self._ui(self._set_progress, pct)

            if _is_importable(import_name):
                self._ui(self._log_append, "✓", "ok",
                         f"{display:<33}  — đã cài")
            else:
                self._ui(self._log_append, "→", "info",
                         f"Đang cài {pip_spec}…", "dim")
                self._ui(self._status_var.set, f"Đang cài {display}…")

                if _pip_install(pip_spec):
                    self._ui(self._log_append, "✓", "ok",
                             f"{display:<33}  — đã cài xong")
                else:
                    self._ui(self._log_append, "✗", "err",
                             f"{display}  — cài thất bại")
                    if "pyopenjtalk" in pip_spec:
                        self._ui(self._log_append, "→", "info",
                                 "Cần Visual C++ Build Tools (Windows)", "dim")
                    elif "style-bert-vits2" in pip_spec:
                        self._ui(self._log_append, "→", "info",
                                 "pip install style-bert-vits2 --no-build-isolation",
                                 "dim")

        # ── VieNeu (PyPI: pip install vieneu) ─────────────────────────────
        url, mod, name = _VIENEU
        self._ui(self._set_progress, 95)
        self._ui(self._status_var.set, f"Kiểm tra {name}…")

        if _is_importable(mod):
            self._ui(self._log_append, "✓", "ok",
                     f"{name:<33}  — đã cài")
        else:
            # Kiểm tra git trước — pip install git+ cần git trên PATH
            if True:  # PyPI — không cần git
                self._ui(self._log_append, "→", "info",
                         f"Đang cài {name} từ PyPI…", "dim")
                self._ui(self._status_var.set, f"Đang cài {name}…")

                ok, err_line = _pip_install_verbose(url)
                if ok:
                    self._ui(self._log_append, "✓", "ok",
                             f"{name:<33}  — đã cài xong")
                else:
                    self._ui(self._log_append, "✗", "err",
                             f"{name}  — cài thất bại")
                    if err_line:
                        self._ui(self._log_append, "→", "info",
                                 err_line, "dim")
                    self._ui(self._log_append, "→", "info",
                             "Xem hướng dẫn: https://github.com/pnnbao97/VieNeu-TTS",
                             "dim")

        # ── Hoàn tất ──────────────────────────────────────────────────────
        self._ui(self._set_progress, 100)
        self._ui(self._log_section, "[ Hoàn tất — đang vào ứng dụng ]")
        self._ui(self._log_append, "✓", "ok", "Khởi động TTS Studio Pro…")
        self._ui(self._status_var.set, "Sẵn sàng!")

        self.success = True
        self._root.after(1200, self._safe_destroy)

    def _safe_destroy(self) -> None:
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    # ─── Public API ────────────────────────────────────────────────────────

    def run(self) -> bool:
        """
        Hiển thị splash screen, kiểm tra / cài đặt thư viện,
        sau đó tự đóng và trả về True.
        """
        threading.Thread(target=self._worker, daemon=True).start()
        self._root.mainloop()
        return self.success


# ─────────────────────────────────────────────────────────────────────────────
# Chạy độc lập để test giao diện
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = SplashScreen().run()
    print(f"\nSplash hoàn tất: {result}")
