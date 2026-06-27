"""
Сравнение детекции людей на LOW vs HI потоках одной камеры.

Захватывает N пар кадров (LOW + HI ~одновременно), запускает YOLO на каждом,
сохраняет side-by-side сравнение и итоговый CSV.

Вывод:
  .output/compare_lohi/<run_id>/
    pair_NNN_low.jpg      — кадр LOW с bbox
    pair_NNN_hi.jpg       — кадр HI с bbox
    pair_NNN_side.jpg     — side-by-side
    results.csv           — детали по каждой паре

Использование:
  python scripts/bench/compare_lohi.py --pairs 20
  python scripts/bench/compare_lohi.py --pairs 50 --trigger-only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.cam_urls import collect_cam_urls, resolve_hi_rtsp_url
from common.utils.motion_utils import (
    StreamReader,
    ffmpeg_capture_options,
    mean_abs_diff,
    open_cap,
    prepare_gray,
    skip_url,
)
from common.utils.time_msk import ts_for_dir

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_MODEL = REPO_ROOT / ".models" / "yolov8n.onnx"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "bench" / "2_compare_lohi"

YOLO_INPUT = 640
PERSON_CLASS = 0


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT / w, YOLO_INPUT / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((YOLO_INPUT, YOLO_INPUT, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT - nw) // 2
    pad_y = (YOLO_INPUT - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_x, pad_y


def _detect(sess, bgr: np.ndarray, conf_threshold: float, nms_threshold: float) -> list[tuple]:
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: blob})[0]
    preds = output[0].T
    scores = preds[:, 4 + PERSON_CLASS]
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    sc = scores[mask]
    bxywh = preds[mask, :4]
    cx, cy, bw, bh = bxywh[:, 0], bxywh[:, 1], bxywh[:, 2], bxywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    indices = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, bw, bh], axis=1).tolist(),
        sc.tolist(), conf_threshold, nms_threshold,
    )
    result = []
    for i in (indices.flatten() if len(indices) else []):
        rx1 = max(0, int((float(x1[i]) - pad_x) / scale))
        ry1 = max(0, int((float(y1[i]) - pad_y) / scale))
        rx2 = min(w, int((float(x1[i]) + float(bxywh[i, 2]) - pad_x) / scale))
        ry2 = min(h, int((float(y1[i]) + float(bxywh[i, 3]) - pad_y) / scale))
        result.append((rx1, ry1, rx2, ry2, float(sc[i])))
    return result


def _draw(bgr: np.ndarray, dets: list) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _side_by_side(left: np.ndarray, right: np.ndarray, label_l: str, label_r: str) -> np.ndarray:
    """Объединяет два кадра рядом, масштабируя к одной высоте."""
    h_target = max(left.shape[0], right.shape[0])
    def _resize_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))
    l = _resize_h(left, h_target)
    r = _resize_h(right, h_target)
    gap = np.zeros((h_target, 4, 3), dtype=np.uint8)
    combined = np.hstack([l, gap, r])
    # Подписи
    for x, label in [(10, label_l), (l.shape[1] + 14, label_r)]:
        cv2.putText(combined, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение LOW vs HI детекции")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=20, help="Количество пар кадров (default: 20)")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--hi-scale", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=6.0,
                        help="Порог diff для захвата пары (0 = захватывать всегда)")
    parser.add_argument("--tcp", action="store_true")
    parser.add_argument("--compare-width", type=int, default=320)
    args = parser.parse_args()

    if not args.env.is_file():
        print(f"Нет .env: {args.env}", file=sys.stderr)
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("Нужен onnxruntime", file=sys.stderr)
        return 1

    if not args.model.is_file():
        print(f"Нет модели: {args.model}", file=sys.stderr)
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

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    cap = open_cap(low_url, **to_ms)
    if cap is None:
        print("LOW не открылся", file=sys.stderr)
        return 1

    reader = StreamReader(hi_url, scale=args.hi_scale, decode_max_fps=1.0, **to_ms)
    reader.start()
    print("Инициализация HI потока...")
    time.sleep(3.0)

    print(f"Камера: {vn}")
    print(f"Пары:   {args.pairs}  threshold={args.threshold}")
    print(f"Вывод:  {out_dir}\n")

    rows = []
    prev_gray = None
    pair_num = 0

    while pair_num < args.pairs:
        ok, low_frame = cap.read()
        if not ok or low_frame is None or low_frame.size == 0:
            time.sleep(0.05)
            continue

        gray = prepare_gray(low_frame, args.compare_width)
        diff = mean_abs_diff(prev_gray, gray) if prev_gray is not None else 0.0
        prev_gray = gray

        if args.threshold > 0 and diff < args.threshold:
            continue

        # Снимаем пару
        hi_frame, _ = reader.get_latest()
        hi_age = reader.age_sec()
        if hi_frame is None or hi_age > 3.0:
            continue

        pair_num += 1
        t_capture = time.time()

        # Детекция на LOW
        dets_low = _detect(sess, low_frame, args.conf, args.nms)
        # Детекция на HI
        dets_hi = _detect(sess, hi_frame, args.conf, args.nms)

        # Сохранение
        low_ann = _draw(low_frame, dets_low)
        hi_ann = _draw(hi_frame, dets_hi)
        side = _side_by_side(
            low_ann, hi_ann,
            f"LOW {low_frame.shape[1]}x{low_frame.shape[0]} det={len(dets_low)}",
            f"HI  {hi_frame.shape[1]}x{hi_frame.shape[0]} det={len(dets_hi)}",
        )

        prefix = f"pair_{pair_num:03d}"
        cv2.imwrite(str(out_dir / f"{prefix}_low.jpg"), low_ann)
        cv2.imwrite(str(out_dir / f"{prefix}_hi.jpg"), hi_ann)
        cv2.imwrite(str(out_dir / f"{prefix}_side.jpg"), side)

        row = {
            "pair": pair_num,
            "diff": round(diff, 2),
            "hi_age_s": round(hi_age, 3),
            "low_w": low_frame.shape[1],
            "low_h": low_frame.shape[0],
            "hi_w": hi_frame.shape[1],
            "hi_h": hi_frame.shape[0],
            "low_dets": len(dets_low),
            "hi_dets": len(dets_hi),
            "low_max_conf": round(max((d[4] for d in dets_low), default=0.0), 3),
            "hi_max_conf": round(max((d[4] for d in dets_hi), default=0.0), 3),
            "agree": "yes" if (len(dets_low) > 0) == (len(dets_hi) > 0) else "no",
        }
        rows.append(row)

        status = f"{'✓' if len(dets_low)==len(dets_hi) else '✗'} LOW={len(dets_low)} HI={len(dets_hi)}"
        print(f"  pair {pair_num:3d}  diff={diff:.2f}  {status}  hi_age={hi_age:.2f}s")

    cap.release()
    reader.stop()

    # CSV
    csv_path = out_dir / "results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Сводка
    agree_count = sum(1 for r in rows if r["agree"] == "yes")
    low_any = sum(1 for r in rows if r["low_dets"] > 0)
    hi_any = sum(1 for r in rows if r["hi_dets"] > 0)
    print(f"\n=== Итог ({len(rows)} пар) ===")
    print(f"  Совпадение LOW/HI: {agree_count}/{len(rows)} ({100*agree_count//max(len(rows),1)}%)")
    print(f"  LOW детектировал: {low_any}/{len(rows)}")
    print(f"  HI  детектировал: {hi_any}/{len(rows)}")
    print(f"  CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
