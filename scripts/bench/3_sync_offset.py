"""
Замер рассинхрона между LOW и HI потоками одной камеры.

Метод: при каждом успешном чтении LOW кадра фиксируем PC-время (monotonic),
затем проверяем возраст последнего HI кадра (age_sec). Разница = задержка HI
относительно LOW.

Дополнительно: если включён --cam-ts, оценивает расхождение OSD-времени
камеры и PC-времени в момент получения кадра.

Вывод:
  .output/sync_offset/<run_id>/
    sync_data.csv
    sync_report.json

Использование:
  python scripts/bench/sync_offset.py --duration 60
  python scripts/bench/sync_offset.py --duration 120 --cam-ts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.cam_urls import collect_cam_urls, resolve_hi_rtsp_url
from common.utils.motion_utils import (
    StreamReader,
    ffmpeg_capture_options,
    open_cap,
    skip_url,
)
from common.utils.time_msk import ts_for_dir

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "bench" / "3_sync_offset"


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер рассинхрона LOW/HI потоков")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--hi-scale", type=float, default=0.5)
    parser.add_argument("--tcp", action="store_true")
    parser.add_argument("--cam-ts", action="store_true",
                        help="Читать OSD-время камеры для оценки camera-PC offset")
    args = parser.parse_args()

    if not args.env.is_file():
        print(f"Нет .env: {args.env}", file=sys.stderr)
        return 1

    from dotenv import load_dotenv
    load_dotenv(args.env, override=True)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=3_000_000
    )

    cameras = collect_cam_urls()
    active = [(k, v) for k, v in cameras if not skip_url(v)]
    if not active:
        print("Нет камер", file=sys.stderr)
        return 1

    vn, low_url = active[0]
    hi_url = resolve_hi_rtsp_url(vn, low_url=low_url, skip_url=skip_url)
    to_ms = dict(open_timeout_ms=10000, read_timeout_ms=10000)

    out_dir = args.output or DEFAULT_OUTPUT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_cap(low_url, **to_ms)
    if cap is None:
        print("LOW не открылся", file=sys.stderr)
        return 1

    reader = StreamReader(
        hi_url, scale=args.hi_scale,
        extract_cam_ts=args.cam_ts,
        **to_ms,
    )
    reader.start()
    print("Инициализация...")
    time.sleep(3.0)

    print(f"Камера: {vn}")
    print(f"Длительность: {args.duration}s  cam_ts={args.cam_ts}")
    print(f"Вывод: {out_dir}\n")

    rows = []
    t_start = time.monotonic()
    t_end = t_start + args.duration
    frame_count = 0
    report_interval = 10.0
    last_report = t_start

    while time.monotonic() < t_end:
        t_before_read = time.monotonic()
        ok, frame = cap.read()
        t_after_read = time.monotonic()

        if not ok or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        frame_count += 1
        read_ms = (t_after_read - t_before_read) * 1000
        hi_age_s = reader.age_sec()
        hi_ts_mono = t_after_read - hi_age_s  # когда HI обновился

        # PC-время сейчас (Unix)
        pc_time = time.time()

        row: dict = {
            "t_rel_s": round(t_after_read - t_start, 3),
            "low_read_ms": round(read_ms, 1),
            "hi_age_s": round(hi_age_s, 4),
            "hi_available": int(hi_age_s < 3.0),
        }

        # OSD время камеры
        if args.cam_ts:
            cam_dt = reader.get_cam_ts()
            if cam_dt is not None:
                cam_unix = cam_dt.timestamp()
                row["cam_pc_offset_s"] = round(pc_time - cam_unix, 1)
                row["cam_time_str"] = cam_dt.strftime("%H:%M:%S")
            else:
                row["cam_pc_offset_s"] = None
                row["cam_time_str"] = None

        rows.append(row)

        # Периодические сообщения
        now = time.monotonic()
        if now - last_report >= report_interval:
            elapsed = now - t_start
            fps_low = frame_count / elapsed
            recent = rows[-50:] if len(rows) >= 50 else rows
            avail_pct = 100 * sum(r["hi_available"] for r in recent) / len(recent)
            age_vals = [r["hi_age_s"] for r in recent if r["hi_available"]]
            mean_age = np.mean(age_vals) if age_vals else 0
            print(f"  t={elapsed:.0f}s  LOW={fps_low:.1f}fps  "
                  f"HI_avail={avail_pct:.0f}%  HI_age_mean={mean_age:.3f}s")
            last_report = now

    cap.release()
    reader.stop()

    # CSV
    csv_path = out_dir / "sync_data.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Статистика
    elapsed = time.monotonic() - t_start
    hi_ages = [r["hi_age_s"] for r in rows if r["hi_available"]]
    low_read_times = [r["low_read_ms"] for r in rows]
    avail_pct = 100 * sum(r["hi_available"] for r in rows) / max(len(rows), 1)

    report = {
        "camera": vn,
        "duration_s": round(elapsed, 1),
        "low_frames": frame_count,
        "low_fps": round(frame_count / elapsed, 2),
        "low_read_mean_ms": round(float(np.mean(low_read_times)), 1) if low_read_times else None,
        "low_read_p95_ms": round(float(np.percentile(low_read_times, 95)), 1) if low_read_times else None,
        "hi_available_pct": round(avail_pct, 1),
        "hi_age_mean_s": round(float(np.mean(hi_ages)), 4) if hi_ages else None,
        "hi_age_p50_s": round(float(np.percentile(hi_ages, 50)), 4) if hi_ages else None,
        "hi_age_p95_s": round(float(np.percentile(hi_ages, 95)), 4) if hi_ages else None,
        "hi_age_max_s": round(float(np.max(hi_ages)), 4) if hi_ages else None,
    }

    if args.cam_ts:
        offsets = [r["cam_pc_offset_s"] for r in rows if r.get("cam_pc_offset_s") is not None]
        if offsets:
            report["cam_pc_offset_mean_s"] = round(float(np.mean(offsets)), 1)
            report["cam_pc_offset_std_s"] = round(float(np.std(offsets)), 1)
            report["cam_pc_offset_min_s"] = round(float(np.min(offsets)), 1)
            report["cam_pc_offset_max_s"] = round(float(np.max(offsets)), 1)

    json_path = out_dir / "sync_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Итог ===")
    print(f"  LOW: {report['low_fps']} fps, read mean={report['low_read_mean_ms']}ms p95={report['low_read_p95_ms']}ms")
    print(f"  HI доступен: {report['hi_available_pct']}%")
    if hi_ages:
        print(f"  HI age: mean={report['hi_age_mean_s']}s  p50={report['hi_age_p50_s']}s  "
              f"p95={report['hi_age_p95_s']}s  max={report['hi_age_max_s']}s")
    if args.cam_ts and "cam_pc_offset_mean_s" in report:
        print(f"  Camera-PC offset: {report['cam_pc_offset_mean_s']}s ± {report['cam_pc_offset_std_s']}s")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
