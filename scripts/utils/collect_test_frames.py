"""
Сбор тестовых кадров из .output/5_diff_yolo_boxes_low для tests/data/.

Структура tests/data/:
  raw_pairs/   — LOW+HI пары (сырые кадры до детекции, имена _low_diff и _raw_diff)
  no_person/   — heartbeat кадры (пустая сцена без людей)
  with_person/ — кадры с людьми (_p1.jpg, _low_p1.jpg)
  artifacts/   — кадры с HEVC-артефактами (очень маленький размер файла)

Использование:
  python scripts/utils/collect_test_frames.py
  python scripts/utils/collect_test_frames.py --output-dir tests/data --max-per-class 20
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / ".output" / "cameras" / "5_diff_yolo_boxes_low"
TEST_DIR = REPO_ROOT / "tests" / "data"

# Максимальный размер файла (байт) для heartbeat/raw — отсекает мусор
_MIN_JPEG_SIZE = 20_000
# Маленький файл — подозрительный (артефакт HEVC или серый кадр)
_ARTIFACT_SIZE_THRESHOLD = 45_000


def collect(output_dir: Path, max_per_class: int) -> None:
    cats: dict[str, list[Path]] = {
        "raw_pairs": [],
        "no_person": [],
        "with_person": [],
        "artifacts": [],
    }

    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        for f in sorted(run_dir.glob("*.jpg")):
            size = f.stat().st_size
            name = f.name

            if "_heartbeat" in name:
                if size < _MIN_JPEG_SIZE:
                    cats["artifacts"].append(f)
                else:
                    cats["no_person"].append(f)
            elif "_low_p" in name or (
                "_p" in name and not "_baseline" in name and not "_heartbeat" in name
                and not "_raw_diff" in name and not "_low_diff" in name
            ):
                cats["with_person"].append(f)

        # raw/ подкаталог: raw_pairs и артефакты
        for f in sorted(run_dir.glob("raw/*.jpg")):
            size = f.stat().st_size
            name = f.name
            if "_low_diff" in name or "_raw_diff" in name:
                if size < _ARTIFACT_SIZE_THRESHOLD:
                    cats["artifacts"].append(f)
                else:
                    cats["raw_pairs"].append(f)

    for cat, files in cats.items():
        dest = output_dir / cat
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        # Берём равномерно по всем запускам
        for f in files:
            if copied >= max_per_class:
                break
            target = dest / f.name
            if not target.exists():
                shutil.copy2(f, target)
                copied += 1
        print(f"  {cat:12s}: скопировано {copied} файлов (всего найдено {len(files)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор тестовых кадров из .output")
    parser.add_argument("--output-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--max-per-class", type=int, default=30,
                        help="Максимум файлов на класс (default: 30)")
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        print(f"Нет каталога: {OUTPUT_DIR}", file=sys.stderr)
        return 1

    print(f"Источник: {OUTPUT_DIR}")
    print(f"Цель:     {args.output_dir}")
    collect(args.output_dir, args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
