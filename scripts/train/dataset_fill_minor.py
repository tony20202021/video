"""
Автодополнение минорных классов из предыдущего датасета.

После всех стратегий: для каждого класса где new_count < prev_count —
копирует все файлы класса из prev_dataset (с пропуском уже существующих).

Usage:
    python scripts/train/dataset_fill_minor.py \\
        .data/groups/v1/dataset \\
        --output .data/groups/v2/dataset

    python scripts/train/dataset_fill_minor.py \\
        .data/groups/v1/dataset \\
        --output .data/groups/v2/dataset \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.log_setup import setup_logging
from common.utils.classes import GROUP_CLASSES

logger = logging.getLogger(__name__)

MSK = timezone(timedelta(hours=3))
_VALID_CLASSES = set(GROUP_CLASSES)

_SOURCES_FIELDS = [
    "timestamp", "strategy", "date", "filename",
    "src_path", "dst_path", "true_class", "pred_class",
    "conf", "conf_2nd", "margin",
]


def _count_jpgs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len(list(path.glob("*.jpg")))


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Дополнение минорных классов: копирует всё из prev, если new < prev"
    )
    parser.add_argument("prev_dataset", type=Path,
                        help="Предыдущий датасет (напр. .data/groups/v1/dataset)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Новый датасет (напр. .data/groups/v2/dataset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Не копировать, только показать что изменилось бы")
    parser.add_argument("--always-fill", action="store_true",
                        help="Копировать все файлы из prev независимо от счётчика new "
                             "(объединение: old ∪ new, дедуп по имени файла)")
    args = parser.parse_args()

    prev_dir = args.prev_dataset.resolve()
    if not prev_dir.is_dir():
        logger.error("Не найден: %s", prev_dir)
        return 1

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== dataset_fill_minor ===")
    logger.info("  prev:   %s", prev_dir)
    logger.info("  output: %s", out_dir)
    logger.info("")

    now_ts = datetime.now(MSK).strftime("%Y%m%d_%H%M%S_msk")
    sources_path = out_dir / "sources.csv"
    write_header = not sources_path.is_file()

    total_copied = total_skipped = 0

    with open(sources_path, "a", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=_SOURCES_FIELDS)
        if write_header:
            sw.writeheader()

        for cls in sorted(_VALID_CLASSES):
            prev_cls_dir = prev_dir / cls
            new_cls_dir = out_dir / cls

            prev_count = _count_jpgs(prev_cls_dir)
            new_count = _count_jpgs(new_cls_dir)

            if prev_count == 0:
                continue

            if not args.always_fill and new_count >= prev_count:
                logger.info("  %s: new=%d >= prev=%d — ок", cls, new_count, prev_count)
                continue

            if args.always_fill:
                logger.info("  %s: new=%d, prev=%d → объединяем (old ∪ new)",
                            cls, new_count, prev_count)
            else:
                need = prev_count - new_count
                logger.info("  %s: new=%d < prev=%d → копируем недостающие (до %d файлов)",
                            cls, new_count, prev_count, need)

            if args.dry_run:
                continue

            new_cls_dir.mkdir(exist_ok=True)
            copied = skipped = 0
            for src in sorted(prev_cls_dir.glob("*.jpg")):
                dst = new_cls_dir / src.name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                    sw.writerow({
                        "timestamp":  now_ts,
                        "strategy":   "F",
                        "date":       prev_dir.name,
                        "filename":   src.name,
                        "src_path":   str(src),
                        "dst_path":   str(dst),
                        "true_class": cls,
                        "pred_class": cls,
                        "conf":       "",
                        "conf_2nd":   "",
                        "margin":     "",
                    })
                else:
                    skipped += 1
            logger.info("    → скопировано: %d  пропущено (уже есть): %d", copied, skipped)
            total_copied += copied
            total_skipped += skipped

    if args.dry_run:
        logger.info("  (dry-run: файлы не скопированы)")
    else:
        logger.info("  Итого: скопировано %d  пропущено %d", total_copied, total_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
