"""Фильтрация кропов жителей.

Пять независимых фильтров:

  1. --min-conf F      — confidence Model 1 из имени файла (_confX.XX).
                         Ловит чужие классы, случайно попавшие в resident.

  2. --min-blur F      — резкость: дисперсия Лапласиана (default: 40).
                         Убирает кадры с motion blur.

  3. --yolo-persons N  — перезапускает YOLO на кропе и считает людей (default: 1).
                         Ловит слияния (p1of1 но на деле 2 человека).
                         Низкий порог (0.12) нужен т.к. камера сверху снижает conf.

  4. --min-coverage F  — доля кадра под bbox человека (default: 0.15).
                         Фильтрует случаи когда YOLO совсем не нашёл человека в кропе.

  5. --min-body F      — доля видимого тела по pose estimation (yolov8n-pose, default: 0).
                         Считает видимые keypoints из 17 (COCO).
                         Ловит "одна рука на весь кадр" — YOLO видит 1 человека,
                         но из 17 точек скелета видно только 1-2 (запястье).
                         Значение 0.3 = видно не менее ~5 из 17 точек.
                         По умолчанию отключён (0) т.к. требует отдельной модели.

Usage:
    python scripts/train/filter_residents.py .data/residents/v0/new --dry-run
    python scripts/train/filter_residents.py .data/residents/v0/new
    python scripts/train/filter_residents.py .data/residents/v0/new --min-body 0.3
    python scripts/train/filter_residents.py .data/residents/v0/new --min-blur 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL      = REPO_ROOT / ".models" / "detect" / "yolov8n.onnx"
DEFAULT_POSE_MODEL = REPO_ROOT / ".models" / "detect" / "yolov8n-pose.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

YOLO_INPUT_SIZE = 640
PERSON_CLASS    = 0

_RE_CONF = re.compile(r"_conf(?P<conf>\d+\.\d+)", re.IGNORECASE)

# cam_01_9_d_YYYYMMDD_HHMMSS_usec_...
_RE_FRAME = re.compile(
    r"^(?P<cam>cam_[^_]+)_\d+_[a-z]_(?P<date>\d{8})_(?P<time>\d{6})_\d+",
    re.IGNORECASE,
)


def _parse_frame(stem: str) -> tuple[str, str, int] | None:
    """Возвращает (cam, date, seconds_of_day) или None."""
    m = _RE_FRAME.match(stem)
    if not m:
        return None
    t = m.group("time")
    sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return m.group("cam"), m.group("date"), sod


# ─── YOLO ─────────────────────────────────────────────────────────────────────

def _load_yolo(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] Нужен onnxruntime: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"[!] Модель не найдена: {model_path}", file=sys.stderr)
        sys.exit(1)
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[np.newaxis], scale, pad_x, pad_y


def _detect_persons(sess, bgr: np.ndarray, conf: float, nms: float
                    ) -> list[tuple[int, int, int, int, float]]:
    """Возвращает список (x1,y1,x2,y2,score) для каждого найденного человека."""
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    output = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    preds = output[0].T
    scores = preds[:, 4 + PERSON_CLASS]
    mask = scores >= conf
    if not mask.any():
        return []
    s = scores[mask]
    bxywh = preds[:, :4][mask]
    bw, bh_ = bxywh[:, 2], bxywh[:, 3]
    x1 = bxywh[:, 0] - bw / 2
    y1 = bxywh[:, 1] - bh_ / 2
    indices = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, bw, bh_], axis=1).tolist(),
        s.tolist(), conf, nms,
    )
    result = []
    for i in (indices.flatten() if len(indices) else []):
        rx1 = int((float(x1[i]) - pad_x) / scale)
        ry1 = int((float(y1[i]) - pad_y) / scale)
        rx2 = int((float(x1[i]) + float(bw[i]) - pad_x) / scale)
        ry2 = int((float(y1[i]) + float(bh_[i]) - pad_y) / scale)
        rx1 = max(0, min(rx1, w - 1))
        ry1 = max(0, min(ry1, h - 1))
        rx2 = max(0, min(rx2, w))
        ry2 = max(0, min(ry2, h))
        result.append((rx1, ry1, rx2, ry2, float(s[i])))
    return result


# ─── Pose estimation ──────────────────────────────────────────────────────────

_pose_model = None

def _load_pose(model_path: Path):
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO as _YOLO
        _pose_model = _YOLO(str(model_path))
    return _pose_model


def _body_coverage(pose_model, bgr: np.ndarray, kp_conf_thresh: float = 0.3) -> float:
    """Доля видимых keypoints из 17 (COCO). 0.0 = ни одной точки, 1.0 = все видны."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = pose_model(bgr, verbose=False)
    if not results or results[0].keypoints is None:
        return 0.0
    kp = results[0].keypoints
    if kp.conf is None or len(kp.conf) == 0:
        return 0.0
    # Берём человека с наибольшим числом видимых точек
    best = 0.0
    for person_kp_conf in kp.conf:
        visible = float((person_kp_conf > kp_conf_thresh).sum())
        total = float(len(person_kp_conf))
        frac = visible / total if total > 0 else 0.0
        best = max(best, frac)
    return best


# ─── Метрики изображения ──────────────────────────────────────────────────────

def _blur_score(bgr: np.ndarray) -> float:
    """Дисперсия Лапласиана. Чем меньше — тем размытее."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _classify_conf(stem: str) -> float | None:
    m = _RE_CONF.search(stem)
    return float(m.group("conf")) if m else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Фильтрация кропов жителей")
    ap.add_argument("dir", type=Path, help="Каталог с кропами (.jpg)")
    ap.add_argument("--min-conf", type=float, default=0.5, metavar="F",
                    help="Мин. confidence Model 1 из имени файла. 0 — отключить. (default: 0.5)")
    ap.add_argument("--min-blur", type=float, default=40.0, metavar="F",
                    help="Мин. резкость (дисперсия Лапласиана). 0 — отключить. (default: 40)")
    ap.add_argument("--yolo-persons", type=int, default=1, metavar="N",
                    help="Макс. людей в кропе по YOLO. 0 — отключить. (default: 1)")
    ap.add_argument("--yolo-conf", type=float, default=0.12, metavar="F",
                    help="Порог уверенности YOLO при детекции в кропе. (default: 0.12)")
    ap.add_argument("--yolo-nms", type=float, default=0.45, metavar="F",
                    help="NMS threshold YOLO. (default: 0.45)")
    ap.add_argument("--min-coverage", type=float, default=0.15, metavar="F",
                    help="Мин. доля кадра под bbox человека (0..1). "
                         "Фильтрует обрывки тела. 0 — отключить. (default: 0.15)")
    ap.add_argument("--min-body", type=float, default=0.0, metavar="F",
                    help="Мин. доля видимых keypoints скелета (0..1). "
                         "0 — отключить (default). Требует yolov8n-pose.pt.")
    ap.add_argument("--pose-kp-conf", type=float, default=0.3, metavar="F",
                    help="Мин. уверенность keypoint чтобы считать его видимым. (default: 0.3)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"Путь к yolov8n.onnx (default: {DEFAULT_MODEL})")
    ap.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL,
                    help=f"Путь к yolov8n-pose.pt (default: {DEFAULT_POSE_MODEL})")
    ap.add_argument("--min-diff", type=float, default=0.0, metavar="F",
                    help="Жадный pixel-diff фильтр: кадр оставляется только если его "
                         "mean-abs-diff от предыдущего оставленного (той же камеры/даты) ≥ F. "
                         "Применяется после остальных фильтров. 0 — отключить (default).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Показать что будет удалено, не удалять")
    args = ap.parse_args()

    images = sorted(f for f in args.dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[!] Нет изображений в {args.dir}", file=sys.stderr)
        return 1

    use_conf     = args.min_conf > 0
    use_blur     = args.min_blur > 0
    use_yolo     = args.yolo_persons > 0 or args.min_coverage > 0
    use_coverage = args.min_coverage > 0
    use_body     = args.min_body > 0
    use_diff     = args.min_diff > 0

    print(f"Каталог:       {args.dir}  ({len(images)} файлов)")
    print(f"Фильтр conf:   {'≥' + str(args.min_conf) if use_conf else 'отключён'}")
    print(f"Фильтр blur:   {'≥' + str(args.min_blur) if use_blur else 'отключён'}")
    print(f"Фильтр YOLO:   {'≤' + str(args.yolo_persons) + ' чел.' if args.yolo_persons > 0 else 'отключён'}"
          + (f", yolo_conf={args.yolo_conf}" if use_yolo else ""))
    print(f"Фильтр cover.: {'≥' + str(args.min_coverage) if use_coverage else 'отключён'}")
    body_str = f"≥{args.min_body:.0%} скелета" if use_body else "отключён"
    print(f"Фильтр body:   {body_str}")
    print(f"Фильтр diff:   {'≥' + str(args.min_diff) if use_diff else 'отключён'}")
    if args.dry_run:
        print("(dry-run — файлы не удаляются)")
    print()

    sess       = _load_yolo(args.model) if use_yolo else None
    pose_model = _load_pose(args.pose_model) if use_body else None

    rm_conf = rm_blur = rm_persons = rm_coverage = rm_body = 0
    surviving: list[Path] = []

    for img_path in images:
        stem = img_path.stem
        reason = None

        # Фильтр 1: confidence из имени файла
        if use_conf and reason is None:
            c = _classify_conf(stem)
            if c is not None and c < args.min_conf:
                reason = f"CONF  {c:.2f} < {args.min_conf}"
                rm_conf += 1

        # Читаем изображение для пикселных фильтров
        bgr = None
        if reason is None and (use_blur or use_yolo or use_body):
            bgr = cv2.imread(str(img_path))

        # Фильтр 2: резкость
        if use_blur and reason is None and bgr is not None:
            blur = _blur_score(bgr)
            if blur < args.min_blur:
                reason = f"BLUR  {blur:.1f} < {args.min_blur}"
                rm_blur += 1

        # Фильтры 3+4: YOLO
        if use_yolo and reason is None and bgr is not None and sess is not None:
            h, w = bgr.shape[:2]
            img_area = h * w
            persons = _detect_persons(sess, bgr, args.yolo_conf, args.yolo_nms)

            if args.yolo_persons > 0 and len(persons) > args.yolo_persons:
                reason = f"MULTI {len(persons)} чел."
                rm_persons += 1
            elif use_coverage and reason is None:
                if not persons:
                    reason = f"COVER 0% (не детектирован)"
                    rm_coverage += 1
                else:
                    best = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    bx1, by1, bx2, by2, _ = best
                    bbox_area = (bx2 - bx1) * (by2 - by1)
                    coverage = bbox_area / img_area if img_area > 0 else 0
                    if coverage < args.min_coverage:
                        reason = f"COVER {coverage:.0%} < {args.min_coverage:.0%}"
                        rm_coverage += 1

        # Фильтр 5: доля тела (pose)
        if use_body and reason is None and pose_model is not None:
            if bgr is None:
                bgr = cv2.imread(str(img_path))
            if bgr is not None:
                frac = _body_coverage(pose_model, bgr, args.pose_kp_conf)
                if frac < args.min_body:
                    reason = f"BODY  {frac:.0%} < {args.min_body:.0%}"
                    rm_body += 1

        if reason is not None:
            print(f"  {reason:<28}  {img_path.name}")
            if not args.dry_run:
                img_path.unlink()
        else:
            surviving.append(img_path)

    # Фильтр 6: жадный pixel-diff (применяется к выжившим после 1–5)
    rm_diff = 0
    if use_diff:
        from collections import defaultdict as _dd
        groups: dict[tuple[str, str], list[tuple[int, Path]]] = _dd(list)
        for f in surviving:
            parsed = _parse_frame(f.stem)
            if parsed is not None:
                cam, date, sod = parsed
                groups[(cam, date)].append((sod, f))
            # файлы без парсинга не участвуют в diff-фильтре → оставляем

        diff_remove: set[Path] = set()
        for (cam, date), items in groups.items():
            items.sort(key=lambda x: x[0])
            last_bgr: np.ndarray | None = None
            for sod, f in items:
                bgr = cv2.imread(str(f))
                if bgr is None:
                    continue
                if last_bgr is None:
                    last_bgr = bgr
                    continue
                h1, w1 = last_bgr.shape[:2]
                h2, w2 = bgr.shape[:2]
                cmp = bgr if (h2 == h1 and w2 == w1) else cv2.resize(bgr, (w1, h1))
                diff = float(np.mean(np.abs(last_bgr.astype(np.float32) - cmp.astype(np.float32))))
                if diff >= args.min_diff:
                    last_bgr = bgr
                else:
                    diff_remove.add(f)

        for f in sorted(diff_remove, key=lambda p: p.name):
            print(f"  {'DIFF  ' + f'{args.min_diff:.0f}':<28}  {f.name}")
            if not args.dry_run:
                f.unlink()
        rm_diff = len(diff_remove)
        surviving = [f for f in surviving if f not in diff_remove]

    kept = len(surviving)

    print()
    print(f"Удалено (conf):     {rm_conf}")
    print(f"Удалено (blur):     {rm_blur}")
    print(f"Удалено (>1 чел.):  {rm_persons}")
    print(f"Удалено (обрывок):  {rm_coverage}")
    print(f"Удалено (тело <):   {rm_body}")
    print(f"Удалено (diff):     {rm_diff}")
    print(f"Оставлено:          {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
