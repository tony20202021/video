"""
Прогоняет GroupClassifier по датасету, создаёт inference-style каталог.

После этого <output>/ можно передать в dataset_v2_from_inference.py —
поддерживаются все стратегии 1,2,3,5,7,8 (как для обычных дат инференса).

Структура вывода:
    <output>/
      <pred_class>/       — файлы (hardlink), где модель предсказала pred_class
      uncertain/          — файлы, где conf < conf_low
      classifications.csv
      labels.json         — true_class из ИСХОДНЫХ каталогов датасета (не из модели)

labels.json содержит пути к файлам внутри <output>/, поэтому
dataset_v2_from_inference.py корректно определяет:
  - стрт 1/7: out_class ≠ true_class (pred ≠ истина)
  - стрт 2:   out_class == uncertain, есть true_class в labels
  - стрт 3:   conf < conf_high, есть true_class в labels
  - стрт 5:   conf >= conf_high, out_class == true_class

Usage:
    python scripts/train/dataset_run_inference.py \\
        .data/groups/v1/dataset \\
        --output .data/groups/v1/inference/images/dataset

    python scripts/train/dataset_run_inference.py \\
        .data/groups/v2/dataset \\
        --output .data/groups/v2/inference/images/dataset \\
        --dry-run

    # Если hardlink невозможен (разные ФС):
    python scripts/train/dataset_run_inference.py ... --copy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.log_setup import setup_logging
from common.utils.classes import GROUP_CLASSES

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
UNCERTAIN_DIR = "uncertain"
_VALID_CLASSES = set(GROUP_CLASSES)

_CSV_FIELDS = [
    "crop", "out_class", "group_conf", "conf_2nd", "margin",
    *[f"p_{cls}" for cls in GROUP_CLASSES],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ef_float(key: str, default: float) -> float:
    val = os.environ.get(key, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


def _load_classifier(config_path: Path):
    if not config_path.is_file():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Ошибка чтения %s: %s", config_path, e)
        return None
    classify_path = cfg.get("models", {}).get("classify")
    if not classify_path:
        logger.warning("config.yaml: нет models.classify")
        return None
    try:
        from ml.classify import GroupClassifier
        clf = GroupClassifier()
        resolved = (
            Path(classify_path) if Path(classify_path).is_absolute()
            else REPO_ROOT / classify_path
        )
        if not clf.load(resolved):
            logger.warning("Не удалось загрузить: %s", resolved)
            return None
        return clf
    except Exception as e:
        logger.warning("Ошибка инициализации GroupClassifier: %s", e)
        return None


def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(str(src), str(dst))
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Прогон GroupClassifier по датасету → inference-style каталог"
    )
    parser.add_argument("dataset_dir", type=Path,
                        help="Исходный датасет (напр. .data/groups/v1/dataset)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Выходной каталог (напр. .data/groups/v1/inference/images/dataset)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conf-low", type=float,
                        default=_ef_float("CLASSIFY_CONF", 0.65),
                        help="Порог uncertain: conf < conf_low → uncertain/ (default: 0.65)")
    parser.add_argument("--copy", action="store_true",
                        help="Копировать файлы вместо hardlink")
    parser.add_argument("--dry-run", action="store_true",
                        help="Не создавать файлы, только статистика")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        logger.error("Не найден: %s", dataset_dir)
        return 1

    clf = _load_classifier(args.config)
    if clf is None or not clf.ready:
        logger.error("GroupClassifier не загружен.")
        return 1

    output_dir = args.output.resolve()

    class_dirs = [
        d for d in sorted(dataset_dir.iterdir())
        if d.is_dir() and d.name in _VALID_CLASSES
    ]
    if not class_dirs:
        logger.warning("Нет каталогов классов в %s", dataset_dir)
        return 0

    total_files = sum(len(list(d.glob("*.jpg"))) for d in class_dirs)
    logger.info("=== dataset_run_inference: %s ===", dataset_dir.name)
    logger.info("  Датасет: %s", dataset_dir)
    logger.info("  Вывод:   %s", output_dir)
    logger.info("  conf_low=%.2f  метод=%s", args.conf_low,
                "copy" if args.copy else "hardlink")
    logger.info("  Классов: %d  Файлов: %d", len(class_dirs), total_files)
    logger.info("")

    import cv2

    csv_rows: list[dict] = []
    labels: dict[str, str] = {}
    # out_class → true_class → count
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    t_start = time.monotonic()
    processed = 0

    for cls_dir in class_dirs:
        true_class = cls_dir.name
        files = sorted(cls_dir.glob("*.jpg"))
        logger.info("  %s: %d файлов", true_class, len(files))

        for jpg in files:
            bgr = cv2.imread(str(jpg))
            if bgr is None:
                continue

            pred_class, conf, prob_map = clf.classify(bgr)
            sorted_probs = sorted(prob_map.values(), reverse=True)
            conf_2nd = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            margin = conf - conf_2nd

            out_class = pred_class if conf >= args.conf_low else UNCERTAIN_DIR
            dst = output_dir / out_class / jpg.name

            # Путь в labels.json = путь назначения (не оригинал),
            # чтобы _collect_from_labels видел out_class как p.parent.name
            labels[str(dst)] = true_class
            stats[out_class][true_class] += 1

            csv_rows.append({
                "crop":       jpg.name,
                "out_class":  out_class,
                "group_conf": round(conf, 4),
                "conf_2nd":   round(conf_2nd, 4),
                "margin":     round(margin, 4),
                **{f"p_{cls}": round(prob_map.get(cls, 0.0), 4)
                   for cls in GROUP_CLASSES},
            })

            if not args.dry_run:
                _link_or_copy(jpg, dst, args.copy)

            processed += 1

    elapsed = time.monotonic() - t_start
    logger.info("")
    logger.info("  Инференс: %d файлов за %.1fс (%.0f мс/файл)",
                processed, elapsed, elapsed / max(processed, 1) * 1000)
    logger.info("")

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_correct = total_wrong = total_uncertain = 0
    for out_cls in sorted(stats):
        breakdown = "  ".join(
            f"{tc}:{n}" for tc, n in sorted(stats[out_cls].items())
        )
        n = sum(stats[out_cls].values())
        if out_cls == UNCERTAIN_DIR:
            logger.info("  → uncertain: %d  [%s]", n, breakdown)
            total_uncertain += n
        else:
            correct = stats[out_cls].get(out_cls, 0)
            wrong = n - correct
            total_correct += correct
            total_wrong += wrong
            logger.info("  → %s: %d  (верно: %d  ошибка: %d)  [%s]",
                        out_cls, n, correct, wrong, breakdown)

    logger.info("")
    acc = 100 * total_correct / max(1, total_correct + total_wrong)
    logger.info("  Итого: %d  точность (без uncertain): %.1f%%  uncertain: %d",
                processed, acc, total_uncertain)

    if args.dry_run:
        logger.info("  (dry-run: файлы не созданы)")
        return 0

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classifications.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)
    logger.info("  classifications.csv → %d строк → %s", len(csv_rows), csv_path)

    # ── Write labels.json ─────────────────────────────────────────────────────
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps({"version": 1, "labels": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  labels.json → %d записей → %s", len(labels), labels_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
