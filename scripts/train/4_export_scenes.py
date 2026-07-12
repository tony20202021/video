"""Экспорт размеченных сцен в датасет, организованный по классам.

Читает labels.json из scene_pool, копирует файлы в выходную директорию
с подпапками по person_id (совместимо с folder-based режимом 5_train_residents.py).

Ключи labels.json могут быть:
  - абсолютным путём (к representatives/scene0042_cam...jpg)
  - basename без префикса (cam...jpg → ищет в pool_dir/)

Usage:
    python scripts/train/4_export_scenes.py \\
        --pool .data/residents/v1/scene_pool \\
        --out  .data/residents/v1/training_export
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_RE_SCENE_PREFIX = re.compile(r"^scene\d+_")

SKIP_LABELS = {"skip"}


def _resolve_src(key: str, pool_dir: Path) -> Path | None:
    """Resolves a label key to an actual file path."""
    p = Path(key)
    if p.is_absolute():
        return p if p.exists() else None
    # bare basename → look in pool_dir directly
    candidate = pool_dir / p.name
    if candidate.exists():
        return candidate
    return None


def export_scenes(pool_dir: Path, out_dir: Path, dry_run: bool = False) -> None:
    labels_file = pool_dir / "labels.json"
    if not labels_file.exists():
        print(f"[!] labels.json not found in {pool_dir}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(labels_file.read_text())
    labels: dict[str, str] = raw.get("labels", raw)

    copied = Counter()
    skipped_label = 0
    missing = 0

    for key, person_id in labels.items():
        if not person_id or person_id in SKIP_LABELS:
            skipped_label += 1
            continue

        src = _resolve_src(key, pool_dir)
        if src is None:
            missing += 1
            continue

        # destination filename: strip scene prefix
        dst_name = _RE_SCENE_PREFIX.sub("", src.name)
        dst = out_dir / person_id / dst_name

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

        copied[person_id] += 1

    total = sum(copied.values())
    print(f"{'[dry-run] ' if dry_run else ''}Экспорт: {pool_dir} → {out_dir}")
    print(f"Скопировано: {total}  |  skip/null: {skipped_label}  |  не найдено: {missing}")
    print("\nПо классам:")
    for cls, cnt in sorted(copied.items()):
        print(f"  {cls}: {cnt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Экспорт сцен в датасет по классам")
    ap.add_argument("--pool", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v1" / "scene_pool",
                    help="Директория scene_pool (содержит labels.json и .jpg)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v1" / "training_export",
                    help="Выходная директория (будет создана)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    export_scenes(args.pool, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
