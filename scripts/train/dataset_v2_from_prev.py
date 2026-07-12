"""
Стратегии 4 и 7: отбор кропов из предыдущего датасета для нового обучения.

  4 — модель колеблется: conf < CLASSIFY_CONF_HIGH (неуверена на известных данных)
  7 — уверенная ошибка: conf ≥ CLASSIFY_CONF_HIGH И predicted ≠ true_class

Запускает GroupClassifier на всех файлах датасета и отбирает
проблемные случаи, которые модель не освоила или освоила неверно.

Usage:
    python scripts/train/dataset_v2_from_prev.py \\
        .data/groups/v1/dataset \\
        --output .data/groups/v2/dataset

    python scripts/train/dataset_v2_from_prev.py \\
        .data/groups/v1/dataset \\
        --output .data/groups/v2/dataset \\
        --strategies 7 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.log_setup import setup_logging
from common.utils.classes import GROUP_CLASSES

logger = logging.getLogger(__name__)

MSK          = timezone(timedelta(hours=3))
_SOURCES_FIELDS = [
    "timestamp", "strategy", "date", "filename",
    "src_path", "dst_path", "true_class", "pred_class",
    "conf", "conf_2nd", "margin",
]

_VALID_CLASSES = set(GROUP_CLASSES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ef_float(key: str, default: float) -> float:
    val = os.environ.get(key, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


def _load_classifier():
    classify_path = os.environ.get("CLASSIFY_MODEL", "").strip()
    if not classify_path:
        logger.warning("CLASSIFY_MODEL не задан в .env")
        return None
    try:
        from ml.classify import GroupClassifier
        clf = GroupClassifier()
        resolved = Path(classify_path) if Path(classify_path).is_absolute() else REPO_ROOT / classify_path
        if not clf.load(resolved):
            logger.warning("Не удалось загрузить: %s", resolved)
            return None
        return clf
    except Exception as e:
        logger.warning("Ошибка инициализации GroupClassifier: %s", e)
        return None


def _apply_quota(candidates: list[dict], max_per_class: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    result = []
    for c in candidates:
        cls = c["true_class"]
        if counts[cls] < max_per_class:
            result.append(c)
            counts[cls] += 1
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Стратегии 4,7: отбор из предыдущего датасета"
    )
    parser.add_argument("dataset_dir", type=Path,
                        help="Предыдущий датасет (напр. .data/groups/v1/dataset)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Выходной каталог нового датасета")
    parser.add_argument("--conf-high",    type=float,
                        default=_ef_float("CLASSIFY_CONF_HIGH", 0.85))
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Квота: максимум файлов на класс")
    parser.add_argument("--strategies",   default="4,7",
                        help="Стратегии через запятую (default: 4,7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Не копировать, только статистика")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        logger.error("Не найден: %s", dataset_dir)
        return 1

    strategies = {int(s.strip()) for s in args.strategies.split(",") if s.strip()}

    clf = _load_classifier()
    if clf is None or not clf.ready:
        logger.error("GroupClassifier не загружен — невозможно запустить инференс.")
        return 1

    logger.info("=== dataset_v2_from_prev: %s ===", dataset_dir.name)
    logger.info("  Стратегии:  %s", sorted(strategies))
    logger.info("  conf_high:  %.2f", args.conf_high)
    logger.info("  Вывод:      %s", args.output)
    logger.info("")

    import cv2

    # ── Scan prev dataset ─────────────────────────────────────────────────────
    class_dirs = [
        d for d in sorted(dataset_dir.iterdir())
        if d.is_dir() and d.name in _VALID_CLASSES
    ]
    if not class_dirs:
        logger.warning("Не найдено каталогов классов в %s", dataset_dir)
        return 0

    total_files = sum(
        len(list(d.glob("*.jpg"))) for d in class_dirs
    )
    logger.info("  Классов: %d  Файлов: %d", len(class_dirs), total_files)

    candidates: list[dict] = []
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
            margin   = conf - conf_2nd

            processed += 1

            # Стратегия 4: модель неуверена
            if 4 in strategies and conf < args.conf_high:
                candidates.append({
                    "strategy":   4,
                    "src":        jpg,
                    "filename":   jpg.name,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "conf":       round(conf, 3),
                    "conf_2nd":   round(conf_2nd, 3),
                    "margin":     round(margin, 3),
                })

            # Стратегия 7: уверенная ошибка
            elif 7 in strategies and conf >= args.conf_high and pred_class != true_class:
                candidates.append({
                    "strategy":   7,
                    "src":        jpg,
                    "filename":   jpg.name,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "conf":       round(conf, 3),
                    "conf_2nd":   round(conf_2nd, 3),
                    "margin":     round(margin, 3),
                })

    elapsed = time.monotonic() - t_start
    logger.info("")
    logger.info("  Инференс: %d файлов за %.1fс (%.0f мс/файл)",
                processed, elapsed, elapsed / max(processed, 1) * 1000)

    # ── Quota ─────────────────────────────────────────────────────────────────
    if args.max_per_class:
        candidates.sort(key=lambda x: x["strategy"])
        before = len(candidates)
        candidates = _apply_quota(candidates, args.max_per_class)
        logger.info("  Квота %d/класс: %d → %d", args.max_per_class, before, len(candidates))

    # ── Stats ─────────────────────────────────────────────────────────────────
    by_strategy: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in candidates:
        by_strategy[c["strategy"]][c["true_class"]] += 1

    for s in sorted(by_strategy):
        total = sum(by_strategy[s].values())
        breakdown = "  ".join(f"{cls}:{n}" for cls, n in sorted(by_strategy[s].items()))
        logger.info("  Стратегия %d: %d  [%s]", s, total, breakdown)
    logger.info("  Итого: %d файлов", len(candidates))

    if args.dry_run or not candidates:
        if args.dry_run:
            logger.info("  (dry-run: файлы не скопированы)")
        return 0

    # ── Copy ──────────────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    sources_path = args.output / "sources.csv"
    write_header = not sources_path.is_file()
    now_ts = datetime.now(MSK).strftime("%Y%m%d_%H%M%S_msk")
    copied = skipped = 0

    with open(sources_path, "a", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=_SOURCES_FIELDS)
        if write_header:
            sw.writeheader()
        for c in candidates:
            cls_dir = args.output / c["true_class"]
            cls_dir.mkdir(exist_ok=True)
            dst = cls_dir / c["filename"]
            if not dst.exists():
                shutil.copy2(str(c["src"]), str(dst))
                copied += 1
            else:
                skipped += 1
            sw.writerow({
                "timestamp":  now_ts,
                "strategy":   c["strategy"],
                "date":       dataset_dir.name,
                "filename":   c["filename"],
                "src_path":   str(c["src"]),
                "dst_path":   str(dst),
                "true_class": c["true_class"],
                "pred_class": c["pred_class"],
                "conf":       c["conf"],
                "conf_2nd":   c["conf_2nd"],
                "margin":     c["margin"],
            })

    logger.info("  Скопировано: %d  Пропущено (уже есть): %d", copied, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
