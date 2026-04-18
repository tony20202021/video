"""
Проверка RTSP-камер из .env (CAM_01_URL … CAM_04_URL).

Читает URL, пытается открыть поток, читает один кадр, собирает свойства OpenCV
(разрешение, fps, backend, fourcc), сохраняет кадры и JSON-отчёт в .output/.

Плейсхолдеры вроде <external_ip> в URL пропускаются.

Примечание: при недоступном RTSP часть сборок OpenCV ждёт открытия потока
до ~30 с — это ограничение backend, не скрипта.

Usage:
    python scripts/verify_cameras.py
    python scripts/verify_cameras.py --env /path/to/.env
    python scripts/verify_cameras.py --tcp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Репозиторий: video/scripts/verify_cameras.py -> video/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output"

CAM_URL_RE = re.compile(r"^CAM_(\d+)_URL$")


def _skip_url(url: str) -> bool:
    u = (url or "").strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    if "<" in u and ">" in u:
        return True
    if not u.lower().startswith("rtsp://"):
        return True
    return False


def _redact_url(url: str) -> str:
    return re.sub(r"(password=)[^&]*", r"\1***", url, flags=re.IGNORECASE)


def _collect_cam_urls() -> list[tuple[str, str]]:
    """Пары (имя_переменной, url), отсортированы по номеру камеры."""
    found: list[tuple[int, str, str]] = []
    for key, val in os.environ.items():
        m = CAM_URL_RE.match(key)
        if not m:
            continue
        found.append((int(m.group(1)), key, val))
    found.sort(key=lambda x: x[0])
    return [(k, v) for _, k, v in found]


def _ffmpeg_capture_options(*, use_tcp: bool, stimeout_us: int) -> str:
    """Опции для OPENCV_FFMPEG_CAPTURE_OPTIONS (разделитель |)."""
    parts: list[str] = [f"stimeout;{stimeout_us}"]
    if use_tcp:
        parts.insert(0, "rtsp_transport;tcp")
    return "|".join(parts)


def probe_stream(
    var_name: str,
    url: str,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> dict:
    import cv2

    result: dict = {
        "env_var": var_name,
        "url_redacted": _redact_url(url),
        "ok": False,
        "opened": False,
        "frame_read": False,
        "error": None,
        "properties": {},
    }

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        try:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
        except Exception:
            pass

        result["opened"] = cap.isOpened()
        if not result["opened"]:
            result["error"] = "VideoCapture.isOpened() == False"
            return result

        props = {
            "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        fourcc_i = int(props["fourcc"])
        if fourcc_i:
            props["fourcc_ascii"] = "".join(
                chr((fourcc_i >> (8 * i)) & 0xFF) for i in range(4)
            )
        backend = cap.getBackendName()
        if backend:
            props["backend"] = str(backend)
        result["properties"] = props

        ok, frame = cap.read()
        result["frame_read"] = bool(ok and frame is not None and frame.size > 0)
        if not result["frame_read"]:
            result["error"] = "cap.read() failed or empty frame"
            return result

        result["ok"] = True
        result["frame_shape"] = list(frame.shape)
        result["_frame_bgr"] = frame
        return result
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка RTSP из .env (CAM_XX_URL)")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="Путь к .env")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Каталог для отчётов (создаётся run_<timestamp>/)",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="RTSP через TCP (rtsp_transport;tcp) — устойчивее через NAT/Wi‑Fi",
    )
    parser.add_argument(
        "--open-timeout-ms",
        type=int,
        default=10000,
        help="Таймаут открытия потока мс (если поддерживается сборкой OpenCV)",
    )
    parser.add_argument(
        "--read-timeout-ms",
        type=int,
        default=10000,
        help="Таймаут чтения кадра мс (если поддерживается)",
    )
    parser.add_argument(
        "--stimeout-us",
        type=int,
        default=8_000_000,
        help="FFmpeg socket I/O timeout (мкс), опция stimeout (default: 8 с)",
    )
    args = parser.parse_args()

    if not args.env.is_file():
        print(f"Файл .env не найден: {args.env}", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Нужен пакет python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1

    load_dotenv(args.env, override=True)

    try:
        import cv2
    except ImportError:
        print("Нужен opencv-python: pip install opencv-python", file=sys.stderr)
        return 1

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options(
        use_tcp=args.tcp,
        stimeout_us=args.stimeout_us,
    )

    cameras = _collect_cam_urls()
    if not cameras:
        print("В .env нет переменных CAM_XX_URL (например CAM_01_URL).", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / f"cam_verify_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_file": str(args.env.resolve()),
        "output_dir": str(run_dir.resolve()),
        "opencv_version": cv2.__version__,
        "rtsp_tcp": bool(args.tcp),
        "ffmpeg_capture_options": os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", ""),
        "cameras": [],
    }

    for var_name, url in cameras:
        entry: dict = {
            "env_var": var_name,
            "skipped": False,
            "skip_reason": None,
        }
        if _skip_url(url):
            entry["skipped"] = True
            entry["skip_reason"] = "пусто, плейсхолдер (<...>) или не rtsp://"
            entry["url_redacted"] = _redact_url(url) if url.strip() else None
            report["cameras"].append(entry)
            print(f"  [-] {var_name}: пропуск ({entry['skip_reason']})")
            continue

        print(f"  … {var_name}: подключение…")
        try:
            probe = probe_stream(
                var_name,
                url,
                open_timeout_ms=args.open_timeout_ms,
                read_timeout_ms=args.read_timeout_ms,
            )
        except Exception as e:
            probe = {
                "env_var": var_name,
                "url_redacted": _redact_url(url),
                "ok": False,
                "opened": False,
                "frame_read": False,
                "error": f"{type(e).__name__}: {e}",
                "properties": {},
            }

        frame = probe.pop("_frame_bgr", None)
        stem = var_name.replace("_URL", "").lower()
        if frame is not None:
            jpg_path = run_dir / f"{stem}_frame.jpg"
            cv2.imwrite(str(jpg_path), frame)
            probe["frame_saved"] = str(jpg_path.name)
        else:
            probe["frame_saved"] = None

        report["cameras"].append(probe)

        status = "OK" if probe.get("ok") else "FAIL"
        print(f"  [{status[0]}] {var_name}: {probe.get('error') or status}")
        if probe.get("properties"):
            p = probe["properties"]
            print(
                f"      размер {p.get('frame_width')}×{p.get('frame_height')}, "
                f"fps {p.get('fps')}, backend {p.get('backend', '—')}"
            )

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {report_path}")

    failed = sum(
        1
        for c in report["cameras"]
        if not c.get("skipped") and not c.get("ok", False)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
