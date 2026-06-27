"""
Оценка детекции людей на датасете .data/.

Структура датасета (два каталога — разметка без JSON):
  .data/
    person/     — кадры с людьми (переместить вручную из no_person)
    no_person/  — кадры без людей (по умолчанию все изображения)

Запуск как часть тестового набора:
    pytest                                      # все тесты
    pytest tests/test_person_detection.py -v    # только этот файл

Запуск как скрипт (полный отчёт + сохранение кадров):
    python tests/test_person_detection.py
    python tests/test_person_detection.py --conf 0.4 --save-fp --save-fn
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / ".models" / "yolov8n.onnx"
DEFAULT_DATA = REPO_ROOT / ".data"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "eval_detection"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Минимально допустимые метрики для прохождения pytest
MIN_PRECISION = 0.5
MIN_RECALL = 0.5


def collect_images(data_dir: Path) -> list[tuple[Path, bool]]:
    items: list[tuple[Path, bool]] = []
    for label, has_person in [("person", True), ("no_person", False)]:
        d = data_dir / label
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, has_person))
    return items


def run_eval(
    model_path: Path = DEFAULT_MODEL,
    data_dir: Path = DEFAULT_DATA,
    conf: float = 0.35,
    nms: float = 0.45,
    save_fp: bool = False,
    save_fn: bool = False,
    save_all: bool = False,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    from common.utils.person_detector import detect_people, draw_boxes, load_model

    sess = load_model(model_path)
    assert sess is not None, f"Не удалось загрузить модель: {model_path}"

    images = collect_images(data_dir)
    assert images, f"Нет изображений в {data_dir}/person/ и {data_dir}/no_person/"

    out_dir = output_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    need_save = save_fp or save_fn or save_all
    if need_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    tp = fp = tn = fn = 0
    errors: list[str] = []

    for img_path, has_person in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        detections = detect_people(sess, bgr, conf_threshold=conf, nms_threshold=nms)
        detected = bool(detections)

        if has_person and detected:
            tp += 1; tag = "TP"
        elif has_person and not detected:
            fn += 1; tag = "FN"
            errors.append(f"FN: {img_path.parent.name}/{img_path.name}")
        elif not has_person and detected:
            fp += 1; tag = "FP"
            confs = [f"{c:.2f}" for *_, c in detections]
            errors.append(f"FP: {img_path.parent.name}/{img_path.name}  conf={confs}")
        else:
            tn += 1; tag = "TN"

        if need_save and (save_all or (save_fp and tag == "FP") or (save_fn and tag == "FN")):
            annotated = draw_boxes(bgr, detections) if detections else bgr
            cv2.imwrite(str(out_dir / f"{tag}_{img_path.stem}.jpg"), annotated)

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "n_person": sum(1 for _, lbl in images if lbl),
        "n_no_person": sum(1 for _, lbl in images if not lbl),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "errors": errors,
        "out_dir": out_dir if need_save else None,
    }


# ─── pytest ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detection_results():
    if not DEFAULT_MODEL.is_file():
        pytest.skip(f"Модель не найдена: {DEFAULT_MODEL} — запустите python scripts/setup_models.py")
    images = collect_images(DEFAULT_DATA)
    if not any(lbl for _, lbl in images):
        pytest.skip("В .data/person/ нет изображений — разметка не задана")
    return run_eval()


def test_precision(detection_results):
    r = detection_results
    print(f"\nPrecision={r['precision']:.3f}  (TP={r['tp']} FP={r['fp']})")
    assert r["precision"] >= MIN_PRECISION, (
        f"Precision {r['precision']:.3f} < {MIN_PRECISION}  FP: {r['fp']}"
    )


def test_recall(detection_results):
    r = detection_results
    print(f"\nRecall={r['recall']:.3f}  (TP={r['tp']} FN={r['fn']})")
    assert r["recall"] >= MIN_RECALL, (
        f"Recall {r['recall']:.3f} < {MIN_RECALL}  FN: {r['fn']}"
    )


def test_no_person_fp_rate(detection_results):
    """Не более 20% ложных срабатываний на кадрах без людей."""
    r = detection_results
    if r["n_no_person"] == 0:
        pytest.skip("Нет изображений в no_person/")
    fp_rate = r["fp"] / r["n_no_person"]
    print(f"\nFP rate={fp_rate:.3f}  (FP={r['fp']} / no_person={r['n_no_person']})")
    assert fp_rate <= 0.20, f"FP rate {fp_rate:.3f} > 0.20"


# ─── standalone ───────────────────────────────────────────────────────────────

def _main() -> int:
    ap = argparse.ArgumentParser(description="Оценка детекции людей на датасете .data/")
    ap.add_argument("--model",  type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data",   type=Path, default=DEFAULT_DATA)
    ap.add_argument("--conf",   type=float, default=0.35)
    ap.add_argument("--nms",    type=float, default=0.45)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--save-fp",  action="store_true")
    ap.add_argument("--save-fn",  action="store_true")
    ap.add_argument("--save-all", action="store_true")
    args = ap.parse_args()

    if not args.model.is_file():
        print(f"Модель не найдена: {args.model}\n  python scripts/setup_models.py", file=sys.stderr)
        return 1

    images = collect_images(args.data)
    if not images:
        print(f"Нет изображений в {args.data}", file=sys.stderr)
        return 1

    try:
        r = run_eval(
            model_path=args.model, data_dir=args.data,
            conf=args.conf, nms=args.nms,
            save_fp=args.save_fp, save_fn=args.save_fn, save_all=args.save_all,
            output_dir=args.output,
        )
    except AssertionError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    for msg in r["errors"]:
        print(f"  {msg}")

    print(f"\n{'─' * 50}")
    print(f"Датасет:   person={r['n_person']}  no_person={r['n_no_person']}  всего={r['total']}")
    print(f"TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")
    print(f"Precision: {r['precision']:.3f}  (мин. {MIN_PRECISION})")
    print(f"Recall:    {r['recall']:.3f}  (мин. {MIN_RECALL})")
    print(f"F1:        {r['f1']:.3f}")
    print(f"Accuracy:  {r['accuracy']:.3f}")
    if r["out_dir"]:
        print(f"Кадры:     {r['out_dir']}")

    ok = r["precision"] >= MIN_PRECISION and r["recall"] >= MIN_RECALL
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
