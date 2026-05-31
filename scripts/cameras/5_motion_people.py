"""
Детекция людей на потоках RTSP: motion detection → YOLOv8n ONNX → сохранение с bounding box.

Два раздельных потока (как в 4_motion_watch):
  LOW (CAM_*_URL)    — субпоток: frame diff для детекции движения
  HI  (CAM_*_HI_URL) — главный поток: захват кадра и YOLO (4x лучшее разрешение)
Если CAM_*_HI_URL не задан — используется тот же LOW-поток.

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls, resolve_hi_rtsp_url
from common.utils.motion_utils import (
    StreamReader,
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    open_cap,
    prepare_gray,
    read_first_plausible_frame,
    redact_url,
    reopen_cap,
    skip_url,
    stem_from_var,
)
from common.utils.time_msk import ts_for_dir, ts_for_file

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "5_motion_people"
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
        description="Motion detection (LOW) + YOLOv8n на HI-потоке: сохранение кадров с людьми"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Путь к yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог confidence (default: 0.35)")
    parser.add_argument("--nms", type=float, default=0.45, help="Порог NMS IoU (default: 0.45)")
    parser.add_argument("--crop-pad", type=float, default=0.10,
                        help="Отступ вокруг bbox при вырезке кропа (доля от bbox, default: 0.10)")
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
    parser.add_argument("--stimeout-us", type=int, default=3_000_000)
    parser.add_argument(
        "--hi-scale", type=float, default=1.0,
        help="Уменьшить HI кадр перед хранением в StreamReader (0.5 = вдвое меньше, ~4× меньше памяти). "
             "Не влияет на DPB FFMPEG, но уменьшает numpy-хранение.",
    )
    parser.add_argument("--min-laplacian-var", type=float, default=12.0)
    parser.add_argument("--min-gray-std", type=float, default=2.5)
    parser.add_argument("--save-extra-reads", type=int, default=12)
    parser.add_argument("--baseline-attempts", type=int, default=16)
    parser.add_argument(
        "--heartbeat-sec", type=float, default=None,
        help="Раз в N сек сохранять кадр с HI независимо от детекции; 0 = выкл; "
             "иначе MOTION_HEARTBEAT_SEC из .env или 600",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Отладка: сохранять сырой HI-кадр при каждом срабатывании порога движения "
             "(до YOLO, в подпапку raw/). Позволяет убедиться, что порог срабатывает.",
    )
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

    if args.heartbeat_sec is not None:
        heartbeat_sec = max(0.0, float(args.heartbeat_sec))
    else:
        raw_hb = (os.environ.get("MOTION_HEARTBEAT_SEC") or "").strip()
        heartbeat_sec = max(0.0, float(raw_hb)) if raw_hb else 600.0

    out_dir = args.output
    if out_dir is None:
        run_id = ts_for_dir()
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

    # HI URL для каждой камеры
    var_low_url: dict[str, str] = {vn: url for vn, url in active}
    var_hi_url: dict[str, str] = {
        vn: resolve_hi_rtsp_url(vn, low_url=url, skip_url=skip_url)
        for vn, url in active
    }

    # LOW потоки — прямые VideoCapture (читаются в основном цикле)
    low_unique = sorted(set(var_low_url.values()))
    caps_by_url: dict[str, cv2.VideoCapture] = {}
    for url in low_unique:
        cap = open_cap(url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
        if cap is None:
            print(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…", file=sys.stderr)
        else:
            caps_by_url[url] = cap

    # HI потоки — StreamReader (фоновые потоки, никогда не блокируют основной цикл)
    hi_unique = sorted(set(var_hi_url.values()))
    hi_readers: dict[str, StreamReader] = {}
    for url in hi_unique:
        reader = StreamReader(
            url,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
            scale=args.hi_scale,
        )
        reader.start()
        hi_readers[url] = reader

    opened_vars: list[tuple[str, str]] = [
        (vn, lu)
        for vn, lu in active
        if lu in caps_by_url and var_hi_url[vn] in hi_readers
    ]
    if not opened_vars:
        print("Ни одна камера не открылась.", file=sys.stderr)
        for c in caps_by_url.values():
            c.release()
        for r in hi_readers.values():
            r.stop()
        return 1

    vars_by_low: dict[str, list[str]] = defaultdict(list)
    for vn, lu in opened_vars:
        vars_by_low[lu].append(vn)

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
    _MAX_LOW_FAILS = 5
    _low_fail: dict[str, int] = defaultdict(int)
    _low_implausible: dict[str, int] = defaultdict(int)
    _MAX_LOW_IMPLAUSIBLE = 40
    # HI переподключение управляется StreamReader автоматически

    import json as _json
    from common.utils.time_msk import ts_iso as _ts_iso
    (out_dir / "run_params.json").write_text(_json.dumps({
        "started_at_msk": _ts_iso(),
        "script": "5_motion_people.py",
        "threshold": threshold,
        "conf": args.conf,
        "nms": args.nms,
        "model": str(args.model),
        "tcp": args.tcp,
        "compare_width": args.compare_width,
        "cameras": [vn for vn, _ in opened_vars],
        "crop_global": list(global_crop) if global_crop else None,
        "heartbeat_sec": heartbeat_sec,
        "output": str(out_dir),
        "dual_stream": True,
        "hi_mode": "StreamReader (non-blocking)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Модель:    {args.model}")
    print(f"Conf:      {args.conf}  NMS: {args.nms}")
    print(f"Threshold: {threshold}  TCP: {args.tcp}")
    print(f"Вывод:     {out_dir}")
    print(f"Камеры ({len(opened_vars)}): {', '.join(vn for vn, _ in opened_vars)}")
    print(f"Обрезка:   {global_crop!r}  [{global_crop_from}]")
    print(f"Потоки:    LOW = прямое чтение | HI = StreamReader (фоновый, неблокирующий)")
    for vn, _ in opened_vars:
        lu, hu = var_low_url[vn], var_hi_url[vn]
        hi_note = "отдельный HI" if hu != lu else "как LOW"
        print(f"  └ {vn}: HI {hi_note}")
    print("Останов: Ctrl+C\n")

    # Базовые кадры: LOW → prev_gray, HI → файл (ждём первый кадр от StreamReader)
    print("Базовый кадр…")
    for low_u, var_list in vars_by_low.items():
        cap_l = caps_by_url[low_u]
        frame_l = read_first_plausible_frame(
            cap_l,
            max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
        )
        if frame_l is None:
            for vn in var_list:
                print(f"  [!] {vn}: нет годного LOW для baseline", file=sys.stderr)
            continue
        for vn in var_list:
            fl = apply_crop_optional(frame_l, crop_by_cam[vn])
            prev_gray[vn] = prepare_gray(fl, args.compare_width)

    by_hi_baseline: dict[str, list[str]] = defaultdict(list)
    for vn, _ in opened_vars:
        by_hi_baseline[var_hi_url[vn]].append(vn)
    for hi_u, vlist in by_hi_baseline.items():
        reader = hi_readers[hi_u]
        deadline = time.monotonic() + 15.0
        fh: np.ndarray | None = None
        while time.monotonic() < deadline:
            fh, _ = reader.get_latest()
            if fh is not None:
                break
            time.sleep(0.2)
        if fh is None:
            for vn in vlist:
                print(f"  [!] {vn}: нет HI baseline за 15с", file=sys.stderr)
            continue
        for vn in vlist:
            im = apply_crop_optional(fh, crop_by_cam[vn])
            stem = stem_from_var(vn)
            ts0 = ts_for_file()
            bname = f"{stem}_{ts0}_baseline.jpg"
            cv2.imwrite(str(out_dir / bname), im)
            print(f"  {bname}")
    print()

    _HI_MAX_AGE = 3.0  # секунд: кадр старше — считаем HI недоступным

    try:
        while True:
            for low_u, var_list in vars_by_low.items():
                # LOW: прямое чтение с автопереподключением
                cap_l = caps_by_url.get(low_u)
                if cap_l is None:
                    if reopen_cap(low_u, caps_by_url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms):
                        print(f"  [R] LOW переподключён: {redact_url(low_u)[:60]}")
                        _low_fail[low_u] = 0
                    continue
                ok_l, frame_l = cap_l.read()
                if not ok_l or frame_l is None or frame_l.size == 0:
                    _low_fail[low_u] += 1
                    if _low_fail[low_u] >= _MAX_LOW_FAILS:
                        print(f"  [!] LOW {_MAX_LOW_FAILS} сбоев — переподключение", file=sys.stderr)
                        reopen_cap(low_u, caps_by_url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
                        _low_fail[low_u] = 0
                        for vn in var_list:
                            prev_gray[vn] = None
                    continue
                _low_fail[low_u] = 0

                # Плохой LOW кадр (HEVC мусор после сна/обрыва) — не считаем diff
                if not frame_decode_plausible(frame_l, min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std):
                    _low_implausible[low_u] += 1
                    if _low_implausible[low_u] >= _MAX_LOW_IMPLAUSIBLE:
                        print(f"  [!] LOW {_MAX_LOW_IMPLAUSIBLE} битых кадров — переподключение", file=sys.stderr)
                        reopen_cap(low_u, caps_by_url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
                        _low_implausible[low_u] = 0
                        for vn in var_list:
                            prev_gray[vn] = None
                    continue
                _low_implausible[low_u] = 0

                # Детекция движения по LOW
                pending_motion: list[tuple[str, float, np.ndarray, np.ndarray]] = []
                for vn in var_list:
                    crop = crop_by_cam[vn]
                    frame_u = apply_crop_optional(frame_l, crop)
                    gray = prepare_gray(frame_u, args.compare_width)
                    prev = prev_gray[vn]
                    if prev is None:
                        prev_gray[vn] = gray
                        continue
                    diff = mean_abs_diff(prev, gray)
                    if diff <= threshold:
                        prev_gray[vn] = gray
                    else:
                        pending_motion.append((vn, diff, gray, frame_u))

                # YOLO на HI-кадре (из StreamReader, мгновенно)
                if pending_motion:
                    by_hi: dict[str, list[tuple[str, float, np.ndarray, np.ndarray]]] = defaultdict(list)
                    for vn, diff, gray_low, frame_low in pending_motion:
                        by_hi[var_hi_url[vn]].append((vn, diff, gray_low, frame_low))

                    for hi_u, entries in by_hi.items():
                        reader = hi_readers.get(hi_u)
                        if reader is None:
                            continue
                        to_save, _ = reader.get_latest()  # не блокирует
                        hi_age = reader.age_sec()
                        if to_save is None or hi_age > _HI_MAX_AGE:
                            for vn, diff, _, frame_low in entries:
                                print(
                                    f"  [~] {vn}: движение diff={diff:.2f}, "
                                    f"HI недоступен (age={hi_age:.1f}s) — YOLO на LOW",
                                    file=sys.stderr,
                                )
                                if args.save_raw:
                                    raw_dir = out_dir / "raw"
                                    raw_dir.mkdir(exist_ok=True)
                                    stem = stem_from_var(vn)
                                    ts_r = ts_for_file()
                                    raw_name = f"{stem}_{ts_r}_low_diff{diff:.2f}.jpg"
                                    cv2.imwrite(str(raw_dir / raw_name), frame_low)
                                    print(f"  [raw-low] {raw_name}")

                                # Fallback: YOLO на LOW-кадре когда HI недоступен
                                detections = detect_people(
                                    sess, frame_low, conf_threshold=args.conf, nms_threshold=args.nms
                                )
                                if not detections:
                                    continue
                                annotated = draw_boxes(frame_low, detections)
                                stem = stem_from_var(vn)
                                ts = ts_for_file()
                                fname = f"{stem}_{ts}_low_p{len(detections)}.jpg"
                                cv2.imwrite(str(out_dir / fname), annotated)
                                h, w = frame_low.shape[:2]
                                crops_dir = out_dir / "crops"
                                crops_dir.mkdir(exist_ok=True)
                                for idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                                    bw, bh = x2 - x1, y2 - y1
                                    px = int(bw * args.crop_pad)
                                    py = int(bh * args.crop_pad)
                                    x1c = max(0, x1 - px)
                                    y1c = max(0, y1 - py)
                                    x2c = min(w, x2 + px)
                                    y2c = min(h, y2 + py)
                                    if x2c > x1c and y2c > y1c:
                                        crop_img = frame_low[y1c:y2c, x1c:x2c]
                                        crop_name = f"{stem}_{ts}_low_p{idx}of{len(detections)}_conf{conf:.2f}.jpg"
                                        cv2.imwrite(str(crops_dir / crop_name), crop_img)
                                print(f"  {fname}  diff={diff:.2f}  люди={len(detections)}  [LOW fallback]")
                            continue

                        if args.save_raw:
                            raw_dir = out_dir / "raw"
                            raw_dir.mkdir(exist_ok=True)

                        for vn, diff, gray_low, frame_low in entries:
                            prev_gray[vn] = gray_low
                            to_c = apply_crop_optional(to_save, crop_by_cam[vn])

                            if args.save_raw:
                                stem = stem_from_var(vn)
                                ts_r = ts_for_file()
                                raw_name = f"{stem}_{ts_r}_raw_diff{diff:.2f}.jpg"
                                cv2.imwrite(str(raw_dir / raw_name), to_c)
                                print(f"  [raw] {raw_name}")

                            detections = detect_people(
                                sess, to_c, conf_threshold=args.conf, nms_threshold=args.nms
                            )
                            if not detections:
                                continue

                            annotated = draw_boxes(to_c, detections)
                            stem = stem_from_var(vn)
                            ts = ts_for_file()
                            fname = f"{stem}_{ts}_p{len(detections)}.jpg"
                            cv2.imwrite(str(out_dir / fname), annotated)

                            h, w = to_c.shape[:2]
                            crops_dir = out_dir / "crops"
                            crops_dir.mkdir(exist_ok=True)
                            for idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                                bw, bh = x2 - x1, y2 - y1
                                px = int(bw * args.crop_pad)
                                py = int(bh * args.crop_pad)
                                x1c = max(0, x1 - px)
                                y1c = max(0, y1 - py)
                                x2c = min(w, x2 + px)
                                y2c = min(h, y2 + py)
                                if x2c > x1c and y2c > y1c:
                                    crop_img = to_c[y1c:y2c, x1c:x2c]
                                    crop_name = f"{stem}_{ts}_p{idx}of{len(detections)}_conf{conf:.2f}.jpg"
                                    cv2.imwrite(str(crops_dir / crop_name), crop_img)

                            print(f"  {fname}  diff={diff:.2f}  люди={len(detections)}")

                # Heartbeat — берём последний кадр из StreamReader
                for vn in var_list:
                    if heartbeat_sec <= 0:
                        continue
                    now = time.monotonic()
                    if now - last_heartbeat[vn] < heartbeat_sec:
                        continue
                    reader = hi_readers.get(var_hi_url[vn])
                    if reader is None:
                        continue
                    hb, _ = reader.get_latest()
                    if hb is None or reader.age_sec() > _HI_MAX_AGE * 2:
                        # Не сбрасываем last_heartbeat — повторим попытку на следующей итерации
                        continue
                    last_heartbeat[vn] = now  # сбрасываем только при успехе
                    hb_u = apply_crop_optional(hb, crop_by_cam[vn])
                    stem = stem_from_var(vn)
                    ts = ts_for_file()
                    hb_name = f"{stem}_{ts}_heartbeat.jpg"
                    cv2.imwrite(str(out_dir / hb_name), hb_u)
                    print(f"  пульс {hb_name}")

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for cap in caps_by_url.values():
            cap.release()
        for reader in hi_readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
