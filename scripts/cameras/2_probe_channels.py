"""
Перебор каналов и стримов камеры XM/iCSee.

Пробует все комбинации channel × stream по RTSP, а также несколько
HTTP-snapshot URL характерных для чипов XM. Сохраняет успешные кадры
и JSON-отчёт в .output/probe_channels_<UTC>/.

Usage:
    # Из .env берёт IP/логин/пароль первой CAM_XX_URL:
    python scripts/cameras/2_probe_channels.py

    # Явные параметры:
    python scripts/cameras/2_probe_channels.py --ip <ip> --user <user> --password <password>

    # Расширить диапазон:
    python scripts/cameras/2_probe_channels.py --channels 0 1 2 3 --streams 0 1

    # Только HTTP-snapshot (без RTSP):
    python scripts/cameras/2_probe_channels.py --http-only

    # RTSP через TCP:
    python scripts/cameras/2_probe_channels.py --tcp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.cam_urls import collect_cam_urls, stem_sort_key  # noqa: E402

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "probe_channels"

DEFAULT_CHANNELS = [0, 1, 2, 3]
DEFAULT_STREAMS   = [0, 1]
RTSP_PORT = 554
HTTP_PORT = 80


# ─── URL builders ─────────────────────────────────────────────────────────────

def rtsp_urls(ip: str, user: str, password: str, channel: int, stream: int) -> list[str]:
    """Все известные форматы RTSP для XM/iCSee."""
    base = f"rtsp://{ip}:{RTSP_PORT}"
    cred = f"user={user}&password={password}" if user else ""
    amp  = "&" if cred else ""
    return [
        # Основной формат iCSee/XM (с ?real_stream)
        f"{base}/{cred}{amp}channel={channel}&stream={stream}.sdp?real_stream",
        # Без ?real_stream
        f"{base}/{cred}{amp}channel={channel}&stream={stream}.sdp",
        # Стандартный RTSP-auth (user:pass@host)
        f"rtsp://{user}:{password}@{ip}:{RTSP_PORT}/channel={channel}&stream={stream}",
    ]


def http_snapshot_urls(ip: str, user: str, password: str, channel: int) -> list[str]:
    """HTTP-snapshot URL для чипов XM."""
    auth = f"{user}:{password}@" if user else ""
    base = f"http://{auth}{ip}"
    return [
        f"{base}/webcapture.jpg?command=snap&channel={channel}",
        f"{base}/snapshot.jpg?channel={channel}",
        f"{base}/cgi-bin/snapshot.cgi?channel={channel}",
        f"{base}/tmpfs/snap.jpg?channel={channel}",
    ]


# ─── Probe helpers ────────────────────────────────────────────────────────────

def probe_rtsp(url: str, *, use_tcp: bool, timeout_ms: int) -> dict:
    import cv2

    if use_tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;tcp|stimeout;{timeout_ms * 1000}"
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"stimeout;{timeout_ms * 1000}"

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return {"ok": False, "error": "not opened"}
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return {"ok": False, "error": "read failed"}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return {"ok": True, "frame": frame, "width": w, "height": h, "fps": fps}
    finally:
        cap.release()


def probe_http(url: str, *, timeout_sec: int) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            data = resp.read()
        if len(data) < 100:
            return {"ok": False, "error": f"too small ({len(data)} bytes)"}
        # Минимальная проверка JPEG-сигнатуры
        if not (data[:2] == b"\xff\xd8" or data[:4] == b"\x89PNG"):
            return {"ok": False, "error": "not JPEG/PNG"}
        return {"ok": True, "data": data, "size": len(data)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── Credentials from .env ────────────────────────────────────────────────────

def _parse_rtsp_creds(url: str) -> tuple[str, str, str] | None:
    """Извлекает (ip, user, password) из RTSP-URL."""
    url = (url or "").strip()
    if not url.lower().startswith("rtsp://") or "<" in url:
        return None
    m = re.search(r"user=([^&]+)&password=([^&]*)", url, re.IGNORECASE)
    if m:
        user, password = m.group(1), m.group(2)
    else:
        m2 = re.match(r"rtsp://([^:@]+):([^@]*)@", url)
        user, password = (m2.group(1), m2.group(2)) if m2 else ("", "")
    m_ip = re.search(r"rtsp://(?:[^@]*@)?([^/:]+)", url)
    ip = m_ip.group(1) if m_ip else ""
    return (ip, user, password) if ip else None


def _extract_from_first_cam_url(env_path: Path) -> tuple[str, str, str] | None:
    """Берёт первую CAM_<stem>_URL через collect_cam_urls(), парсит credentials."""
    try:
        from dotenv import dotenv_values
        import os
        vals = dotenv_values(env_path)
        old = {k: os.environ.get(k) for k in vals}
        os.environ.update({k: v for k, v in vals.items() if v is not None})
        cameras = collect_cam_urls()
        # восстанавливаем переменные окружения
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    except ImportError:
        return None

    for _, url in cameras:
        creds = _parse_rtsp_creds(url)
        if creds:
            return creds
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Перебор channel/stream камеры XM/iCSee")
    ap.add_argument("--ip",       default=None, help="IP камеры")
    ap.add_argument("--user",     default="",   help="Логин")
    ap.add_argument("--password", default="",   help="Пароль")
    ap.add_argument("--channels", nargs="+", type=int, default=DEFAULT_CHANNELS,
                    metavar="N", help=f"Каналы для проверки (default: {DEFAULT_CHANNELS})")
    ap.add_argument("--streams",  nargs="+", type=int, default=DEFAULT_STREAMS,
                    metavar="N", help=f"Стримы (default: {DEFAULT_STREAMS})")
    ap.add_argument("--tcp",  action="store_true", help="RTSP через TCP")
    ap.add_argument("--http-only", action="store_true", help="Только HTTP-snapshot, без RTSP")
    ap.add_argument("--rtsp-only", action="store_true", help="Только RTSP, без HTTP")
    ap.add_argument("--timeout-ms",  type=int, default=6000,  help="Таймаут RTSP мс (default: 6000)")
    ap.add_argument("--timeout-http",type=int, default=5,     help="Таймаут HTTP сек (default: 5)")
    ap.add_argument("--env",    type=Path, default=DEFAULT_ENV)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # Загружаем credentials из .env если не заданы явно
    if not args.ip and args.env.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env, override=False)
            creds = _extract_from_first_cam_url(args.env)
            if creds:
                args.ip, args.user, args.password = creds
                print(f"Credentials из .env: IP={args.ip} user={args.user}")
            else:
                print(f"Не удалось извлечь IP из {args.env}", file=sys.stderr)
        except ImportError:
            print("python-dotenv не установлен: pip install python-dotenv", file=sys.stderr)

    if not args.ip:
        print(
            "Укажите --ip, или добавьте в .env переменную вида CAM_01_URL=rtsp://IP:554/...",
            file=sys.stderr,
        )
        print("Пример запуска:", file=sys.stderr)
        print("  python scripts/cameras/2_probe_channels.py --ip <ip> --user <user> --password <password>", file=sys.stderr)
        return 1

    if not args.http_only:
        try:
            import cv2
        except ImportError:
            print("Нужен opencv-python: pip install opencv-python", file=sys.stderr)
            return 1

    run_id  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / f"probe_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    ok_count = 0

    # ── RTSP ──────────────────────────────────────────────────────────────────
    if not args.http_only:
        import cv2
        print(f"\nRTSP  {args.ip}  channels={args.channels}  streams={args.streams}  tcp={args.tcp}")
        print("─" * 64)
        for ch in args.channels:
            for st in args.streams:
                urls = rtsp_urls(args.ip, args.user, args.password, ch, st)
                for url in urls:
                    redacted = re.sub(r"(password=)[^&/]*", r"\1***", url)
                    print(f"  ch={ch} stream={st}  {redacted} … ", end="", flush=True)
                    r = probe_rtsp(url, use_tcp=args.tcp, timeout_ms=args.timeout_ms)
                    entry = {
                        "proto": "rtsp",
                        "channel": ch,
                        "stream": st,
                        "url_redacted": redacted,
                        **{k: v for k, v in r.items() if k != "frame"},
                    }
                    if r["ok"]:
                        fname = f"rtsp_ch{ch}_st{st}.jpg"
                        cv2.imwrite(str(run_dir / fname), r["frame"])
                        entry["frame_saved"] = fname
                        ok_count += 1
                        print(f"OK  {r['width']}×{r['height']} {r['fps']:.1f}fps  → {fname}")
                        results.append(entry)
                        break  # нашли рабочий URL для этой комбинации ch/st
                    else:
                        print(r["error"])
                        results.append(entry)

    # ── HTTP snapshot ──────────────────────────────────────────────────────────
    if not args.rtsp_only:
        print(f"\nHTTP snapshot  {args.ip}  channels={args.channels}")
        print("─" * 64)
        for ch in args.channels:
            urls = http_snapshot_urls(args.ip, args.user, args.password, ch)
            for url in urls:
                redacted = re.sub(r"(://)[^:@/]+:[^@/]+@", r"\1***:***@", url)
                print(f"  ch={ch}  {redacted} … ", end="", flush=True)
                r = probe_http(url, timeout_sec=args.timeout_http)
                entry = {
                    "proto": "http",
                    "channel": ch,
                    "url_redacted": redacted,
                    **{k: v for k, v in r.items() if k != "data"},
                }
                if r["ok"]:
                    fname = f"http_ch{ch}.jpg"
                    (run_dir / fname).write_bytes(r["data"])
                    entry["frame_saved"] = fname
                    ok_count += 1
                    print(f"OK  {r['size']} bytes  → {fname}")
                    results.append(entry)
                    break  # нашли рабочий URL для этого канала
                else:
                    print(r["error"])
                    results.append(entry)

    # ── Отчёт ─────────────────────────────────────────────────────────────────
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ip": args.ip,
        "user": args.user,
        "channels_tested": args.channels,
        "streams_tested":  args.streams,
        "rtsp_tcp": args.tcp,
        "results": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'─' * 64}")
    print(f"Успешных: {ok_count} / {len(results)}")
    print(f"Отчёт:   {report_path}")
    if ok_count:
        print(f"Кадры:   {run_dir}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
