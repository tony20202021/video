"""Сбор кропов жителей для датасета Модели 2.

Копирует файлы класса 1_resident из датасета групп и накопленного инференса
в очередь нового датасета жителей (.data/residents/v1/new/).

Источники по умолчанию:
  .data/groups/v1/dataset/1_resident/
  .data/groups/v1/inference/images/*/1_resident/

Дедупликация по имени файла — файл с тем же именем копируется только один раз.

Usage:
    python scripts/train/collect_residents.py
    python scripts/train/collect_residents.py --out .data/residents/v1/new
    python scripts/train/collect_residents.py --src .data/groups/v1/dataset/1_resident
    python scripts/train/collect_residents.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
RESIDENT_CLASS = "1_resident"


def _default_sources() -> list[Path]:
    base = REPO_ROOT / ".data" / "groups" / "v1"
    sources: list[Path] = []
    dataset_dir = base / "dataset" / RESIDENT_CLASS
    if dataset_dir.is_dir():
        sources.append(dataset_dir)
    inference_root = base / "inference" / "images"
    if inference_root.is_dir():
        for date_dir in sorted(inference_root.iterdir()):
            d = date_dir / RESIDENT_CLASS
            if d.is_dir():
                sources.append(d)
    return sources


def collect(
    sources: list[Path],
    out_dir: Path,
    *,
    dry_run: bool = False,
    ext: set[str] = IMAGE_EXTS,
) -> tuple[int, int]:
    """Копирует файлы из sources в out_dir. Возвращает (скопировано, пропущено)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = {f.name for f in out_dir.iterdir() if f.is_file()}

    copied = skipped = 0
    for src_dir in sources:
        if not src_dir.is_dir():
            print(f"  [!] Не найдено: {src_dir}", file=sys.stderr)
            continue
        files = sorted(f for f in src_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in ext)
        for f in files:
            if f.name in seen:
                skipped += 1
                continue
            seen.add(f.name)
            if not dry_run:
                shutil.copy2(f, out_dir / f.name)
            copied += 1

    return copied, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Сбор кропов жителей для датасета Модели 2")
    ap.add_argument("--src", type=Path, action="append", dest="sources", metavar="DIR",
                    help="Каталог с кропами 1_resident (можно передавать несколько раз). "
                         "По умолчанию: dataset/1_resident/ + inference/images/*/1_resident/")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v1" / "new",
                    help="Куда копировать (default: .data/residents/v1/new/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Показать что будет скопировано, не копировать")
    args = ap.parse_args()

    sources = args.sources or _default_sources()
    if not sources:
        print("[!] Источники не найдены.", file=sys.stderr)
        return 1

    print(f"Источники ({len(sources)}):")
    for s in sources:
        n = sum(1 for f in s.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS) if s.is_dir() else 0
        print(f"  {s}  [{n} файлов]")
    print(f"Назначение: {args.out}")
    if args.dry_run:
        print("(dry-run — файлы не копируются)")
    print()

    copied, skipped = collect(sources, args.out, dry_run=args.dry_run)

    print(f"Скопировано: {copied}  Пропущено (дубли): {skipped}")
    total = sum(1 for f in args.out.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS) if args.out.is_dir() else 0
    print(f"Итого в {args.out}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
