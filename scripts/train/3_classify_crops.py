"""
Запуск классификации групп на каталоге с кропами людей.

Читает кропы из crops/ или любого каталога, прогоняет через GroupClassifier,
выводит сводку и сохраняет results.csv.

Если модель не обучена — все возвращают «unknown» с conf=0.0 (ожидаемо).

Использование:
  python scripts/train/classify_crops.py
  python scripts/train/classify_crops.py --input .output/cameras/5_diff_yolo_boxes_low/run_XXX/crops
  python scripts/train/classify_crops.py --model models/classify/v1.onnx --conf 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.classify import GroupClassifier, CLASSES

DEFAULT_MODEL = REPO_ROOT / ".models" / "classify" / "v0.onnx"
DEFAULT_INPUT  = REPO_ROOT / ".output" / "cameras" / "5_diff_yolo_boxes_low"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "train" / "3_classify_crops"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_crops(input_path: Path) -> list[Path]:
    """Собирает кропы: если input_path — каталог с crops/, берём оттуда, иначе ищем рекурсивно."""
    crops_dir = input_path / "crops" if (input_path / "crops").is_dir() else input_path
    files = sorted(f for f in crops_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if not files and input_path.is_dir():
        # Ищем во всех run_* подкаталогах
        for run_dir in sorted(input_path.iterdir()):
            sub_crops = run_dir / "crops"
            if sub_crops.is_dir():
                files.extend(sorted(f for f in sub_crops.iterdir() if f.suffix.lower() in IMAGE_EXTS))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Классификация кропов людей")
    parser.add_argument("--input",  type=Path, default=None,
                        help="Каталог с кропами или run_* каталог (default: .output/cameras/5_diff_yolo_boxes_low)")
    parser.add_argument("--model",  type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf",   type=float, default=0.0,
                        help="Минимальная confidence для показа (default: 0 = показывать всё)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Путь для results.csv")
    parser.add_argument("--max",    type=int,   default=200, help="Максимум файлов (default: 200)")
    args = parser.parse_args()

    clf = GroupClassifier(args.model)
    if not clf.ready:
        print(f"Модель не загружена: {args.model}", file=sys.stderr)
        print("  Если модель необучена — результаты будут 'unknown'. Это нормально.", file=sys.stderr)
    else:
        print(f"Модель: {args.model}")

    input_dir = args.input or DEFAULT_INPUT
    crops = collect_crops(input_dir)[:args.max]
    if not crops:
        print(f"Нет кропов в {input_dir}", file=sys.stderr)
        return 1

    print(f"Кропов: {len(crops)}\n")

    rows = []
    class_counts: dict[str, int] = {c: 0 for c in CLASSES + ["unknown"]}

    for path in crops:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        cls, conf, probs = clf.classify(bgr)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        row = {
            "file": path.name,
            "class": cls,
            "conf": round(conf, 3),
            **{c: round(probs.get(c, 0.0), 3) for c in CLASSES},
        }
        rows.append(row)
        if conf >= args.conf:
            print(f"  {path.name[:50]:50s}  {cls:12s}  conf={conf:.3f}")

    print(f"\n=== Распределение ({len(rows)} кропов) ===")
    for cls in CLASSES + ["unknown"]:
        cnt = class_counts.get(cls, 0)
        if cnt > 0:
            print(f"  {cls:12s}: {cnt:4d} ({100*cnt//max(len(rows),1)}%)")

    if rows:
        from common.utils.time_msk import ts_for_dir
        run_dir = args.output or (DEFAULT_OUTPUT / f"run_{ts_for_dir()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "classify_results.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
