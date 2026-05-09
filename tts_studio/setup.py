"""
Setup script — TTS Studio Pro
Chạy: python setup.py
"""

import sys
import subprocess
import shutil
import shlex
import platform
from pathlib import Path

# Fix UTF-8 output on Windows (cp1252 terminal)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_PYTHON = (3, 10)
ROOT = Path(__file__).parent

PACKAGES = {
    "audio": [
        ("numpy>=1.24.0",              "numpy"),
        ("soundfile>=0.12.0",          "soundfile"),
        ("pydub>=0.25.0",              "pydub"),
    ],
    "language_detection": [
        ("langdetect>=1.0.9",          "langdetect"),
        ("lingua-language-detector>=2.0.0", "lingua"),
    ],
    "engine_en": [
        ("kokoro>=0.9.0",              "kokoro"),
    ],
    "engine_ja": [
        ("style-bert-vits2",           "style_bert_vits2"),
        ("pyopenjtalk",                "pyopenjtalk"),
    ],
    "utilities": [
        ("tqdm>=4.65.0",               "tqdm"),
    ],
}

# VieNeu không có trên PyPI — hướng dẫn riêng
VIENEU_INSTALL_GUIDE = """
  Cài VieNeu-TTS từ PyPI:
    pip install vieneu
  Windows (tránh lỗi build llama-cpp):
    pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
  Xem thêm: https://github.com/pnnbao97/VieNeu-TTS
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"
CYAN  = "\033[96m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def section(title): print(f"\n{BOLD}{title}{RESET}")


def run_pip(package_spec: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *shlex.split(package_spec),
         "--quiet", "--disable-pip-version-check"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def is_importable(module_name: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True
    )
    return result.returncode == 0


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python() -> bool:
    section("[ 1 / 4 ]  Kiểm tra Python")
    ver = sys.version_info
    if ver >= MIN_PYTHON:
        ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")
        return True
    err(f"Python {ver.major}.{ver.minor} — cần >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    info("Tải Python mới hơn tại: https://www.python.org/downloads/")
    return False


def check_system_deps():
    section("[ 2 / 4 ]  Kiểm tra system dependencies")
    if check_ffmpeg():
        ok("ffmpeg đã có trên PATH")
    else:
        warn("ffmpeg chưa cài — xuất MP3 sẽ không khả dụng (fallback WAV)")
        os_name = platform.system()
        if os_name == "Windows":
            info("Cài ffmpeg: winget install Gyan.FFmpeg")
            info("hoặc tải tại: https://ffmpeg.org/download.html")
        elif os_name == "Darwin":
            info("Cài ffmpeg: brew install ffmpeg")
        else:
            info("Cài ffmpeg: sudo apt install ffmpeg")


def install_packages():
    section("[ 3 / 4 ]  Cài đặt thư viện Python")
    results = {}

    for group, pkgs in PACKAGES.items():
        print(f"\n  [{group}]")
        for pip_spec, import_name in pkgs:
            if is_importable(import_name):
                ok(f"{pip_spec}  (đã có)")
                results[pip_spec] = True
            else:
                info(f"Đang cài {pip_spec} ...")
                success = run_pip(pip_spec)
                if success:
                    ok(f"{pip_spec}")
                else:
                    err(f"{pip_spec}  — cài thất bại")
                    if "pyopenjtalk" in pip_spec:
                        info("pyopenjtalk cần Visual C++ Build Tools (Windows)")
                        info("Tải: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                    elif "style-bert-vits2" in pip_spec:
                        info("Thử: pip install style-bert-vits2 --no-build-isolation")
                results[pip_spec] = success

    # VieNeu — có trên PyPI, tự động cài
    print(f"\n  [engine_vi]")
    if is_importable("vieneu"):
        ok("vieneu  (đã có)")
    else:
        import platform as _plat
        pip_spec = (
            "vieneu --extra-index-url "
            "https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/"
            if _plat.system() == "Windows"
            else "vieneu"
        )
        info(f"Đang cài {pip_spec} ...")
        success = run_pip(pip_spec)
        if success:
            ok("vieneu")
        else:
            err("vieneu  — cài thất bại")
            print(VIENEU_INSTALL_GUIDE)

    return results


def check_engine_status():
    section("[ 4 / 4 ]  Tình trạng engine")

    engines = [
        ("kokoro",           "🇺🇸 Tiếng Anh  (Kokoro TTS)"),
        ("vieneu",           "🇻🇳 Tiếng Việt (VieNeu-TTS)"),
        ("style_bert_vits2", "🇯🇵 Tiếng Nhật (Style-Bert-VITS2)"),
    ]

    available = []
    for module, label in engines:
        if is_importable(module):
            ok(f"{label}")
            available.append(label)
        else:
            warn(f"{label}  — chưa khả dụng")

    return available


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  TTS Studio Pro — Setup{RESET}")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.executable}")
    print(f"{BOLD}{'='*60}{RESET}")

    if not check_python():
        sys.exit(1)

    check_system_deps()
    install_packages()
    available = check_engine_status()

    print(f"\n{BOLD}{'='*60}{RESET}")
    if available:
        print(f"{GREEN}{BOLD}  Setup hoàn tất!{RESET}")
        print(f"  Engine khả dụng: {', '.join(available)}")
        print(f"\n  Khởi động ứng dụng:")
        print(f"    cd {ROOT}")
        print(f"    python main.py")
    else:
        print(f"{YELLOW}{BOLD}  Setup xong nhưng chưa có engine nào khả dụng.{RESET}")
        print(f"  Kiểm tra lại lỗi cài đặt ở trên.")
    print(f"{BOLD}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
