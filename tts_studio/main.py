"""
TTS Studio Pro — Entry Point

Usage:
    python main.py
    python main.py --port 9000
    python main.py --no-browser
    python main.py --config path/to/config.json
"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

_HERE = Path(__file__).parent
_VERSION = "1.0.0"

# ANSI colours (stripped on Windows if not supported)
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"

def _enable_ansi():
    """Enable ANSI escape codes on Windows 10+."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Engine availability check
# ─────────────────────────────────────────────────────────────────────────────

_ENGINES = [
    ("en", "kokoro",           "🇺🇸 Tiếng Anh  (Kokoro TTS)"),
    ("vi", "vieneu",           "🇻🇳 Tiếng Việt (VieNeu-TTS)"),
    ("ja", "style_bert_vits2", "🇯🇵 Tiếng Nhật (Style-Bert-VITS2)"),
]

_INSTALL_HINTS = {
    "en": "pip install kokoro>=0.9.0",
    "vi": "pip install vieneu",
    "ja": "pip install style-bert-vits2",
}


def check_engines() -> dict[str, bool]:
    """
    Try importing each TTS engine module.
    Returns {lang: available} without loading model weights.
    """
    import importlib
    status = {}
    for lang, module, _ in _ENGINES:
        try:
            importlib.import_module(module)
            status[lang] = True
        except ImportError:
            status[lang] = False
    return status


def print_engine_report(status: dict[str, bool]) -> None:
    for lang, module, label in _ENGINES:
        ok = status.get(lang, False)
        if ok:
            print(f"  {_GREEN}✓{_RESET}  {label}")
        else:
            hint = _INSTALL_HINTS.get(lang, "")
            print(f"  {_YELLOW}⚠{_RESET}  {label}  {_DIM}— não khả dụng{_RESET}")
            print(f"       {_DIM}→ {hint}{_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(port: int, available: list[str]) -> None:
    lang_icons = {"en": "🇺🇸", "vi": "🇻🇳", "ja": "🇯🇵"}
    icons = " ".join(lang_icons[l] for l in available if l in lang_icons)
    url   = f"http://localhost:{port}"

    print()
    print(f"{_BOLD}{'─' * 52}{_RESET}")
    print(f"{_BOLD}  TTS Studio Pro  v{_VERSION}{_RESET}")
    print(f"{'─' * 52}")
    print(f"  {_CYAN}URL      {_RESET} {_BOLD}{url}{_RESET}")
    print(f"  {_CYAN}Port     {_RESET} {port}")
    print(f"  {_CYAN}Engines  {_RESET} {icons if icons else '(none)'}")
    print(f"  {_CYAN}Platform {_RESET} Python {sys.version.split()[0]}  "
          f"{sys.platform}")
    print(f"{'─' * 52}")
    print(f"  {_DIM}Press Ctrl+C to quit{_RESET}")
    print(f"{_BOLD}{'─' * 52}{_RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tts_studio",
        description="TTS Studio Pro — Multilingual TTS (EN / VI / JA)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=None,
        help="HTTP port (default: from config, usually 8766)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open a browser window on startup",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        metavar="PATH",
        help="Path to a JSON config file (overrides config.py defaults)",
    )
    parser.add_argument(
        "--no-splash", action="store_true",
        help="Bỏ qua màn hình khởi động (dùng khi đã cài đủ thư viện)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _enable_ansi()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    sys.path.insert(0, str(_HERE))
    from config import Config
    cfg = Config.load(args.config)

    # CLI flags override config
    port       = args.port if args.port is not None else cfg.PORT
    open_browser = cfg.OPEN_BROWSER and not args.no_browser

    # ── Màn hình khởi động ─────────────────────────────────────────────────
    if not getattr(args, "no_splash", False):
        try:
            from splash import SplashScreen
            SplashScreen().run()
            import importlib
            importlib.invalidate_caches()   # nhận diện gói vừa cài
        except ImportError:
            pass   # tkinter không khả dụng, bỏ qua splash

    # ── Engine check ───────────────────────────────────────────────────────
    print(f"\n{_BOLD}[ 1 / 3 ]  Engine check{_RESET}")
    engine_status = check_engines()
    print_engine_report(engine_status)
    available = [lang for lang, ok in engine_status.items() if ok]

    if not available:
        print(f"\n  {_RED}✗  No TTS engine available.{_RESET}")
        print(f"  Run:  {_CYAN}python setup.py{_RESET}  to install engines.")
        sys.exit(1)

    # ── Start server ───────────────────────────────────────────────────────
    print(f"\n{_BOLD}[ 2 / 3 ]  Starting HTTP server on port {port}{_RESET}")
    from api.server import start_server

    ui_dir     = _HERE / "ui"
    export_dir = Path(cfg.DEFAULT_EXPORT)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        server = start_server(
            port=port,
            ui_dir=ui_dir,
            export_dir=export_dir,
        )
    except OSError as exc:
        print(f"  {_RED}✗  Cannot bind to port {port}: {exc}{_RESET}")
        print(f"  Try a different port:  python main.py --port 8767")
        sys.exit(1)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"  {_GREEN}✓{_RESET}  Listening on http://localhost:{port}")

    # ── Open browser ───────────────────────────────────────────────────────
    print(f"\n{_BOLD}[ 3 / 3 ]  Launching browser{_RESET}")
    url = f"http://localhost:{port}"
    if open_browser:
        try:
            webbrowser.open(url)
            print(f"  {_GREEN}✓{_RESET}  Opened {url}")
        except Exception as exc:
            print(f"  {_YELLOW}⚠{_RESET}  Could not open browser: {exc}")
            print(f"  Open manually: {_BOLD}{url}{_RESET}")
    else:
        print(f"  {_DIM}(browser launch skipped — --no-browser){_RESET}")
        print(f"  Open manually: {_BOLD}{url}{_RESET}")

    # ── Banner + wait ──────────────────────────────────────────────────────
    print_banner(port, available)

    try:
        server_thread.join()          # block until server thread exits
    except KeyboardInterrupt:
        print(f"\n  {_DIM}Shutting down…{_RESET}")
        server.shutdown()
        print(f"  {_GREEN}✓{_RESET}  Bye!")


if __name__ == "__main__":
    main()
