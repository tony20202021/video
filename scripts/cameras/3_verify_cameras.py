"""
Проверка RTSP-камер из .env: переменные вида **CAM_<идентификатор>_URL** (напр. CAM_01_URL, CAM_01_9_U_URL, CAM_01_10_D_URL).

Читает URL, пытается открыть поток, читает один кадр, собирает свойства OpenCV
(разрешение, fps, backend, fourcc), сохраняет кадры и JSON-отчёт в `.output/cam_verify/`.

Плейсхолдеры вроде <external_ip> в URL пропускаются.

Обрезка склеенного кадра: **MOTION_CROP_REL**, **CAM_<stem>_CROP_REL** (как в 4_motion_diff_low), опционально **--crop-rel**.
В отчёт и JPEG попадает кадр **после обрезки**; в JSON — crop_rel и размер до/после.

Примечание: при недоступном RTSP часть сборок OpenCV ждёт открытия потока
до ~30 с — это ограничение backend, не скрипта.

Usage:
    python scripts/cameras/3_verify_cameras.py
    python scripts/cameras/3_verify_cameras.py --env /path/to/.env
    python scripts/cameras/3_verify_cameras.py --tcp
    python scripts/cameras/3_verify_cameras.py --crop-rel 0,0,1,0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Репозиторий: video/scripts/cameras/3_verify_cameras.py -> video/
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls as _collect_cam_urls
from common.utils.motion_utils import redact_url as _redact_url
from common.utils.time_msk import ts_for_dir, ts_iso

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "cameras" / "3_cam_verify"


def _check_rtsp_port(host: str, port: int = 554, timeout: float = 2.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _extract_host(url: str) -> str | None:
    import re
    m = re.search(r"rtsp://(?:[^@]*@)?([^/:]+)", url)
    return m.group(1) if m else None


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
            host = _extract_host(url)
            port_open = _check_rtsp_port(host) if host else None
            if port_open is False:
                cause = "порт 554 недоступен — камера offline или сеть недоступна"
            elif port_open is True:
                cause = "порт 554 открыт, RTSP не открылся — возможно камера занята другим клиентом или неверные credentials"
            else:
                cause = "не удалось определить хост"
            result["error"] = "VideoCapture.isOpened() == False"
            result["possible_cause"] = cause
            result["port_554_open"] = port_open
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
    parser = argparse.ArgumentParser(
        description="Проверка RTSP из .env (CAM_<stem>_URL, напр. CAM_01_9_U_URL)"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="Путь к .env")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Родительский каталог (по умолчанию .output/cam_verify); внутри создаётся cam_verify_<UTC>/",
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
    parser.add_argument(
        "--retry-sec",
        type=float,
        default=0.0,
        help="Если камера занята — ждать N секунд и повторить (0 = не повторять)",
    )
    parser.add_argument(
        "--no-hi",
        action="store_true",
        help="Не проверять CAM_*_HI_URL (по умолчанию HI-потоки включены)",
    )
    parser.add_argument(
        "--crop-rel",
        type=str,
        default=None,
        metavar="X,Y,W,H",
        help="Глобальная обрезка x,y,w,h (доли 0…1), перебивает MOTION_CROP_REL из .env",
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
        print(
            "В .env нет переменных CAM_<stem>_URL (например CAM_01_URL, CAM_01_9_U_URL).",
            file=sys.stderr,
        )
        return 1

    # Добавляем HI-потоки если не передан --no-hi.
    # Сохраняем маппинг hi_key → low_var_name чтобы передать правильную обрезку:
    # HI-поток снимает тот же склеенный кадр что и LOW, обрезка одинакова.
    hi_to_low_var: dict[str, str] = {}
    if not args.no_hi:
        from common.utils.cam_urls import companion_hi_url_env_key
        hi_entries: list[tuple[str, str]] = []
        for var_name, low_url in cameras:
            hi_key = companion_hi_url_env_key(var_name)
            hi_url = (os.environ.get(hi_key) or "").strip()
            if hi_url and not _skip_url(hi_url):
                hi_entries.append((hi_key, hi_url))
                hi_to_low_var[hi_key] = var_name  # HI наследует crop от LOW
        cameras = cameras + hi_entries

    active_for_crop = [(k, v) for k, v in cameras if not _skip_url(v)]
    crop_cli = (args.crop_rel or "").strip() or None
    global_crop, global_crop_from = resolve_global_crop(
        crop_rel_arg=crop_cli,
        motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    if crop_cli and global_crop is None:
        return 1
    crop_by_cam = crop_map_for_cameras(active_for_crop, global_crop=global_crop)

    run_id = ts_for_dir()
    run_dir = args.output / f"cam_verify_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at_msk": ts_iso(),
        "env_file": str(args.env.resolve()),
        "output_dir": str(run_dir.resolve()),
        "opencv_version": cv2.__version__,
        "rtsp_tcp": bool(args.tcp),
        "ffmpeg_capture_options": os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", ""),
        "crop_global": list(global_crop) if global_crop else None,
        "crop_global_from": global_crop_from,
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

        attempts = 0
        max_attempts = 2 if args.retry_sec > 0 else 1
        while True:
            attempts += 1
            suffix = f" (попытка {attempts})" if attempts > 1 else ""
            print(f"  … {var_name}: подключение…{suffix}")
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
            if probe.get("ok") or attempts >= max_attempts:
                break
            cause = probe.get("possible_cause", "")
            if "занята" in cause or "busy" in cause.lower():
                import time as _time
                print(f"  [!] {var_name}: камера занята, жду {args.retry_sec:.0f} с…")
                _time.sleep(args.retry_sec)
            else:
                break

        frame = probe.pop("_frame_bgr", None)
        stem = var_name.replace("_URL", "").lower()
        if frame is not None:
            # HI-поток: обрезка берётся от соответствующего LOW (та же физическая камера)
            crop_key = hi_to_low_var.get(var_name, var_name)
            crop = crop_by_cam.get(crop_key)
            if crop is not None:
                src_shape = list(frame.shape)
                frame = apply_crop_optional(frame, crop)
                probe["crop_rel"] = list(crop)
                probe["frame_shape_before_crop"] = src_shape
                probe["frame_shape"] = list(frame.shape)
            else:
                probe["crop_rel"] = None
            jpg_path = run_dir / f"{stem}_frame.jpg"
            cv2.imwrite(str(jpg_path), frame)
            probe["frame_saved"] = str(jpg_path.name)
        else:
            probe["frame_saved"] = None
            crop_key = hi_to_low_var.get(var_name, var_name)
            probe["crop_rel"] = crop_by_cam.get(crop_key)
            if probe["crop_rel"] is not None:
                probe["crop_rel"] = list(probe["crop_rel"])

        report["cameras"].append(probe)

        status = "OK" if probe.get("ok") else "FAIL"
        print(f"  [{status[0]}] {var_name}: {probe.get('error') or status}")
        if probe.get("properties"):
            p = probe["properties"]
            print(
                f"      размер {p.get('frame_width')}×{p.get('frame_height')}, "
                f"fps {p.get('fps')}, backend {p.get('backend', '—')}"
            )
        if probe.get("crop_rel") is not None:
            print(f"      обрезка x,y,w,h={probe['crop_rel']} → кадр {probe.get('frame_shape')}")

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
