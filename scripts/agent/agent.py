"""
Локальный агент: читает RTSP, детектирует людей, отправляет события на backend.

Запускается на каждом локальном роутере/машине рядом с камерами.
Не требует прямого доступа к MongoDB — общается только с backend по HTTP.

Usage:
    python scripts/agent/agent.py
    python scripts/agent/agent.py --backend http://my-server:8780 --threshold 4
    python scripts/agent/agent.py --heartbeat-sec 300
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls
from common.utils.motion_utils import (
    drain_cap_buffer,
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    open_cap,
    pick_frame_to_save,
    prepare_gray,
    read_first_plausible_frame,
    skip_url,
    stem_from_var,
)
from common.utils.person_detector import detect_people, load_model
from common.utils.time_msk import ts_for_file, ts_iso

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_MODEL = REPO_ROOT / ".models" / "detect" / "yolov8n.onnx"
DEFAULT_CONFIG = REPO_ROOT / "agents.yaml"


def _load_host_config(config_path: Path, host: str) -> dict:
    """Загружает блок конфига для указанного хоста из agents.yaml."""
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[!] Нужен PyYAML: pip install pyyaml", file=sys.stderr)
        return {}
    if not config_path.is_file():
        print(f"[!] Файл конфига не найден: {config_path}", file=sys.stderr)
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hosts = cfg.get("hosts", {})
    if host not in hosts:
        available = ", ".join(hosts.keys())
        print(f"[!] Хост '{host}' не найден в конфиге. Доступные: {available}", file=sys.stderr)
        return {}
    return hosts[host]


def _apply_host_config(host_cfg: dict) -> None:
    """Устанавливает переменные окружения из блока конфига хоста."""
    if not host_cfg:
        return
    # Камеры
    for key, val in host_cfg.get("cameras", {}).items():
        os.environ.setdefault(key, str(val))
    # Параметры движения
    motion = host_cfg.get("motion", {})
    for env_key, cfg_key in [
        ("MOTION_DIFF_THRESHOLD", "threshold"),
        ("MOTION_HEARTBEAT_SEC", "heartbeat_sec"),
        ("BACKEND_URL", "backend_url"),
    ]:
        val = host_cfg.get("backend_url") if cfg_key == "backend_url" else motion.get(cfg_key)
        if val is not None:
            os.environ.setdefault(env_key, str(val))


def _frame_to_b64(frame) -> str:
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def _send_event(
    backend_url: str,
    camera_id: str,
    diff: float,
    detections: list,
    frame,
    crops: list,
    agent_host: str,
) -> bool:
    """POST /ingest/event на backend. Возвращает True при успехе."""
    try:
        import urllib.request
        import urllib.error
        import json as _json

        frame_b64 = _frame_to_b64(frame)
        crops_b64 = [_frame_to_b64(c) for c in crops]

        payload = {
            "camera_id": camera_id,
            "timestamp_msk": ts_iso(),
            "diff": round(diff, 2),
            "detections": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 3)}
                for x1, y1, x2, y2, conf in detections
            ],
            "frame_b64": frame_b64,
            "crops_b64": crops_b64,
            "agent_host": agent_host,
        }
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{backend_url.rstrip('/')}/ingest/event",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  [!] Ошибка отправки: {e}", file=sys.stderr)
        return False


def _check_backend(backend_url: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{backend_url.rstrip('/')}/ingest/ping", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description="Локальный агент камер → backend")
    ap.add_argument("--host", default=None,
                    help="Имя блока хоста из agents.yaml (напр. home, server)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="Путь к agents.yaml (default: agents.yaml в корне репо)")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--crop-pad", type=float, default=0.10)
    ap.add_argument("--heartbeat-sec", type=float, default=None)
    ap.add_argument("--tcp", action="store_true")
    ap.add_argument("--compare-width", type=int, default=320)
    ap.add_argument("--open-timeout-ms", type=int, default=10000)
    ap.add_argument("--read-timeout-ms", type=int, default=10000)
    ap.add_argument("--stimeout-us", type=int, default=8_000_000)
    ap.add_argument("--min-laplacian-var", type=float, default=12.0)
    ap.add_argument("--min-gray-std", type=float, default=2.5)
    ap.add_argument("--save-extra-reads", type=int, default=12)
    ap.add_argument("--baseline-attempts", type=int, default=16)
    args = ap.parse_args()

    # Загружаем конфиг хоста из agents.yaml (до .env чтобы .env мог перебить)
    if args.host:
        host_cfg = _load_host_config(args.config, args.host)
        _apply_host_config(host_cfg)
        # TCP из конфига если не задан явно
        motion_cfg = host_cfg.get("motion", {})
        if not args.tcp and motion_cfg.get("tcp"):
            args.tcp = True
        if args.conf is None or args.conf == 0.35:
            args.conf = float(motion_cfg.get("conf", args.conf or 0.35))
        if args.nms is None or args.nms == 0.45:
            args.nms = float(motion_cfg.get("nms", args.nms or 0.45))

    if args.env.is_file():
        from dotenv import load_dotenv
        load_dotenv(args.env, override=True)

    # backend из аргумента или env
    if not args.backend:
        args.backend = os.environ.get("BACKEND_URL", "http://localhost:8780")

    # Параметры из env
    threshold = args.threshold or float(os.environ.get("MOTION_DIFF_THRESHOLD") or 10.0)
    hb_raw = os.environ.get("MOTION_HEARTBEAT_SEC") or ""
    heartbeat_sec = args.heartbeat_sec if args.heartbeat_sec is not None else (
        float(hb_raw) if hb_raw else 600.0
    )

    # Проверяем backend
    print(f"Backend: {args.backend}")
    if not _check_backend(args.backend):
        print("[!] Backend недоступен — работаем в автономном режиме (без отправки)")

    # Модель
    sess = load_model(args.model)
    if sess is None:
        print(f"Модель не найдена: {args.model}", file=sys.stderr)
        return 1

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us
    )

    cameras = collect_cam_urls()
    active = [(k, v) for k, v in cameras if not skip_url(v)]
    if not active:
        print("Нет активных CAM_*_URL", file=sys.stderr)
        return 1

    crop_cli = None
    global_crop, _ = resolve_global_crop(
        crop_rel_arg=crop_cli,
        motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    crop_by_cam = crop_map_for_cameras(active, global_crop=global_crop)

    caps: dict[str, cv2.VideoCapture] = {}
    for var_name, url in active:
        cap = open_cap(url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
        if cap is None:
            print(f"  [!] {var_name}: не удалось открыть", file=sys.stderr)
            continue
        caps[var_name] = cap

    if not caps:
        print("Ни одна камера не открылась.", file=sys.stderr)
        return 1

    prev_gray: dict[str, object] = {k: None for k in caps}
    last_heartbeat: dict[str, float] = {k: time.monotonic() for k in caps}

    import socket
    agent_host = socket.gethostname()

    print(f"Threshold: {threshold}  Heartbeat: {heartbeat_sec}s  Cameras: {list(caps)}")
    print("Базовый кадр…")
    for var_name, cap in caps.items():
        f0 = read_first_plausible_frame(
            cap, max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std,
        )
        if f0 is not None:
            crop = crop_by_cam[var_name]
            f0c = apply_crop_optional(f0, crop)
            prev_gray[var_name] = prepare_gray(f0c, args.compare_width)
            print(f"  baseline: {var_name}")
    print()

    try:
        while True:
            for var_name, cap in list(caps.items()):
                drain_cap_buffer(cap)
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    continue

                crop = crop_by_cam[var_name]
                frame_c = apply_crop_optional(frame, crop)
                gray = prepare_gray(frame_c, args.compare_width)

                # Heartbeat
                if heartbeat_sec > 0:
                    now = time.monotonic()
                    if now - last_heartbeat[var_name] >= heartbeat_sec:
                        last_heartbeat[var_name] = now
                        if frame_decode_plausible(
                            frame_c,
                            min_laplacian_var=args.min_laplacian_var,
                            min_gray_std=args.min_gray_std,
                        ):
                            _send_event(
                                args.backend, stem_from_var(var_name),
                                diff=0.0, detections=[], frame=frame_c, crops=[],
                                agent_host=agent_host,
                            )
                            print(f"  пульс {var_name}")

                prev = prev_gray[var_name]
                if prev is None:
                    prev_gray[var_name] = gray
                    continue

                diff = mean_abs_diff(prev, gray)
                prev_gray[var_name] = gray
                if diff <= threshold:
                    continue

                # Движение → YOLO
                to_check = pick_frame_to_save(
                    cap, frame,
                    max_extra_reads=args.save_extra_reads,
                    min_laplacian_var=args.min_laplacian_var,
                    min_gray_std=args.min_gray_std,
                )
                if to_check is None:
                    continue

                to_c = apply_crop_optional(to_check, crop)
                detections = detect_people(sess, to_c, conf_threshold=args.conf, nms_threshold=args.nms)
                if not detections:
                    continue

                # Кропы с padding
                h, w = to_c.shape[:2]
                crops = []
                for x1, y1, x2, y2, conf in detections:
                    bw, bh = x2 - x1, y2 - y1
                    px, py = int(bw * args.crop_pad), int(bh * args.crop_pad)
                    x1c, y1c = max(0, x1 - px), max(0, y1 - py)
                    x2c, y2c = min(w, x2 + px), min(h, y2 + py)
                    if x2c > x1c and y2c > y1c:
                        crops.append(to_c[y1c:y2c, x1c:x2c])

                ok_send = _send_event(
                    args.backend, stem_from_var(var_name),
                    diff=diff, detections=detections, frame=to_c, crops=crops,
                    agent_host=agent_host,
                )
                status_mark = "→" if ok_send else "✗"
                print(f"  {status_mark} {stem_from_var(var_name)}  diff={diff:.1f}  люди={len(detections)}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nОстанов.")
    finally:
        for cap in caps.values():
            cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
