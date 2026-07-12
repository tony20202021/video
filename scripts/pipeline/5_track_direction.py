"""CLI-обёртка трекинга направления движения.

Usage:
    python scripts/pipeline/5_track_direction.py \
        .output/pipeline/2_yolo_boxes_files/run_20260629_XXX
    python scripts/pipeline/5_track_direction.py run_XXX --config zones.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.track_direction import (
    DIRECTION_HOME, DIRECTION_AWAY, DIRECTION_UNKNOWN,
    run_tracking,
)
from common.utils.log_setup import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "pipeline" / "5_track_direction"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Трекинг людей и определение направления движения"
    )
    ap.add_argument("input_dir", type=Path,
                    help="run_*-каталог из 2_yolo_boxes_files (содержит detections.csv)")
    ap.add_argument("--config", type=Path, default=None,
                    help="Путь к YAML с настройками зон и трекера (опционально)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    setup_logging()

    if not args.input_dir.is_dir():
        logger.warning("[!] Не найдено: %s", args.input_dir)
        return 1

    out_dir = args.output or (DEFAULT_OUTPUT / f"run_{args.input_dir.name}")

    logger.info("Вход:   %s", args.input_dir)
    logger.info("Вывод:  %s", out_dir)

    results = run_tracking(args.input_dir, args.config, out_dir)

    if not results:
        logger.info("Треков не найдено.")
        return 0

    by_dir: dict[str, int] = defaultdict(int)
    for r in results:
        by_dir[r["direction"]] += 1

    logger.info("Итого треков: %s", len(results))
    labels = {DIRECTION_HOME: "→ домой", DIRECTION_AWAY: "← из дома",
              DIRECTION_UNKNOWN: "неизвестно"}
    for d, n in sorted(by_dir.items()):
        logger.info("  %s: %s", labels.get(d, d), n)
    logger.info("Вывод: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
