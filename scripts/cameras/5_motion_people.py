"""
Детекция людей на потоках RTSP: motion detection → YOLOv8n ONNX → сохранение с bounding box.

При обнаружении движения запускает минимальную модель (YOLOv8n ONNX).
Сохраняет кадры только если обнаружен хотя бы один человек (класс 0 COCO).
На сохраняемый кадр накладываются прямоугольники с confidence.

Модель: models/yolov8n.onnx (скачать: https://github.com/ultralytics/assets/releases)
  wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx

Usage:
    python scripts/cameras/5_motion_people.py
    python scripts/cameras/5_motion_people.py --model models/yolov8n.onnx --conf 0.4
    python scripts/cameras/5_motion_people.py --tcp --threshold 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls
from common.utils.motion_utils import (
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

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "motion_people"
DEFAULT_MODEL = REPO_ROOT / "models" / "yolov8n.onnx"

YOLO_INPUT_SIZE = 640
PERSON_CLASS = 0
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─── YOLOv8n inference ────────────────────────────────────────────────────────

def _load_model(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("Нужен onnxruntime: pip install onnxruntime", file=sys.stderr)
        return None
    if not model_path.is_file():
        print(
            f"Модель не найдена: {model_path}\n"
            f"  Скачать: wget -P models/ "
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
            file=sys.stderr,
        )
        return None
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Возвращает (blob, scale, pad_x, pad_y) для letterbox-ресайза в 640×640."""
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
    blob = blob.transpose(2, 0, 1)[np.newaxis]            # HWC→NCHW
    return blob, scale, pad_x, pad_y


def _postprocess(
    output: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    nms_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Возвращает список (x1, y1, x2, y2, confidence) для людей в координатах orig."""
    # output shape: [1, 84, 8400] → transpose → [8400, 84]
    preds = output[0].T  # [8400, 84]
    # 84 = cx, cy, w, h + 80 class scores
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    person_scores = class_scores[:, PERSON_CLASS]

    mask = person_scores >= conf_threshold
    if not mask.any():
        return []

    scores = person_scores[mask]
    bxywh = boxes_xywh[mask]

    # cx,cy,w,h в координатах 640×640 → x1,y1,x2,y2
    cx, cy, bw, bh = bxywh[:, 0], bxywh[:, 1], bxywh[:, 2], bxywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2

    boxes_for_nms = np.stack([x1, y1, bw, bh], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_threshold, nms_threshold)

    result = []
    for i in (indices.flatten() if len(indices) else []):
        # обратный letterbox: убираем паддинг, делим на scale
        rx1 = int((float(x1[i]) - pad_x) / scale)
        ry1 = int((float(y1[i]) - pad_y) / scale)
        rx2 = int((float(x1[i]) + float(bxywh[i, 2]) - pad_x) / scale)
        ry2 = int((float(y1[i]) + float(bxywh[i, 3]) - pad_y) / scale)
        rx1 = max(0, min(rx1, orig_w - 1))
        ry1 = max(0, min(ry1, orig_h - 1))
        rx2 = max(0, min(rx2, orig_w))
        ry2 = max(0, min(ry2, orig_h))
        result.append((rx1, ry1, rx2, ry2, float(scores[i])))
    return result


def detect_people(
    sess,
    bgr: np.ndarray,
    *,
    conf_threshold: float,
    nms_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: blob})[0]
    return _postprocess(
        output,
        orig_w=w,
        orig_h=h,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )


def draw_boxes(bgr: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 2), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Motion detection + YOLOv8n: сохранение кадров с людьми"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Путь к yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог confidence (default: 0.35)")
    parser.add_argument("--nms", type=float, default=0.45, help="Порог NMS IoU (default: 0.45)")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Порог mean abs diff движения; иначе MOTION_DIFF_THRESHOLD из .env или 10",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp", action="store_true", help="RTSP через TCP")
    parser.add_argument("--crop-rel", type=str, default=None, metavar="X,Y,W,H")
    parser.add_argument("--compare-width", type=int, default=320)
    parser.add_argument("--open-timeout-ms", type=int, default=10000)
    parser.add_argument("--read-timeout-ms", type=int, default=10000)
    parser.add_argument("--stimeout-us", type=int, default=8_000_000)
    parser.add_argument("--min-laplacian-var", type=float, default=12.0)
    parser.add_argument("--min-gray-std", type=float, default=2.5)
    parser.add_argument("--save-extra-reads", type=int, default=12)
    parser.add_argument("--baseline-attempts", type=int, default=16)
    args = parser.parse_args()

    if not args.env.is_file():
        print(f"Файл .env не найден: {args.env}", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Нужен python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1

    load_dotenv(args.env, override=True)

    sess = _load_model(args.model)
    if sess is None:
        return 1

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        raw_t = (os.environ.get("MOTION_DIFF_THRESHOLD") or "").strip()
        threshold = float(raw_t) if raw_t else 10.0

    out_dir = args.output
    if out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us
    )

    cameras = collect_cam_urls()
    active = [(k, v) for k, v in cameras if not skip_url(v)]
    if not active:
        print("Нет активных rtsp:// URL.", file=sys.stderr)
        return 1

    crop_cli = (args.crop_rel or "").strip() or None
    global_crop, global_crop_from = resolve_global_crop(
        crop_rel_arg=crop_cli,
        motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    if crop_cli and global_crop is None:
        return 1
    crop_by_cam = crop_map_for_cameras(active, global_crop=global_crop)

    print(f"Модель:    {args.model}")
    print(f"Conf:      {args.conf}  NMS: {args.nms}")
    print(f"Threshold: {threshold}  TCP: {args.tcp}")
    print(f"Вывод:     {out_dir}")
    print(f"Камеры:    {', '.join(k for k, _ in active)}")
    print(f"Обрезка:   {global_crop!r}  [{global_crop_from}]")
    print("Останов: Ctrl+C\n")

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

    prev_gray: dict[str, np.ndarray | None] = {k: None for k in caps}

    print("Базовый кадр…")
    for var_name, cap in caps.items():
        frame0 = read_first_plausible_frame(
            cap,
            max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
        )
        if frame0 is None:
            print(f"  [!] {var_name}: нет годного кадра для baseline", file=sys.stderr)
            continue
        crop = crop_by_cam[var_name]
        frame0c = apply_crop_optional(frame0, crop)
        stem = stem_from_var(var_name)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(out_dir / f"{stem}__{ts}_baseline.jpg"), frame0c)
        prev_gray[var_name] = prepare_gray(frame0c, args.compare_width)
        print(f"  baseline: {stem}")
    print()

    try:
        while True:
            for var_name, cap in list(caps.items()):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    continue
                crop = crop_by_cam[var_name]
                frame_c = apply_crop_optional(frame, crop)
                gray = prepare_gray(frame_c, args.compare_width)
                prev = prev_gray[var_name]
                if prev is None:
                    prev_gray[var_name] = gray
                    continue

                diff = mean_abs_diff(prev, gray)
                prev_gray[var_name] = gray
                if diff <= threshold:
                    continue

                # Движение — проверяем кадр и запускаем детектор
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

                annotated = draw_boxes(to_c, detections)
                stem = stem_from_var(var_name)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                fname = f"{stem}_{ts}_p{len(detections)}.jpg"
                cv2.imwrite(str(out_dir / fname), annotated)
                print(f"  {fname}  diff={diff:.2f}  люди={len(detections)}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for cap in caps.values():
            cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
