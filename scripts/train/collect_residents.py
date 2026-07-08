"""Сбор кропов жителей для датасета Модели 2.

Копирует файлы класса 1_resident из датасета групп и накопленного инференса
в очередь нового датасета жителей (.data/residents/v1/new/).

Источники по умолчанию:
  .data/groups/v1/dataset/1_resident/
  .data/groups/v1/inference/images/*/1_resident/

Дедупликация и фильтрация:
  1. По имени файла — файл с тем же именем копируется только один раз.
  2. labels.json — если рядом с папкой-источником (inference/images/DATE/) есть labels.json
     и файл переразмечен в класс вне collect_classes → пропускается.
  3. Фильтр по числу людей (--max-persons N) — отбрасывает кадры где в оригинале
     было больше N человек. Парсит суффикс _pXofN_. По умолчанию: 1 (только одиночные).
  4. Временна́я (--interval N) — один кадр на (камера, дата, N-секундное окно).
     Парсит имя вида cam_01_9_d_YYYYMMDD_HHMMSS_usec_...
     При конфликте берёт файл с наибольшим conf (из суффикса _confX.XX).

Usage:
    python scripts/train/collect_residents.py
    python scripts/train/collect_residents.py --out .data/residents/v1/new
    python scripts/train/collect_residents.py --src .data/groups/v1/dataset/1_resident
    python scripts/train/collect_residents.py --interval 10
    python scripts/train/collect_residents.py --max-persons 1   # только одиночные кадры
    python scripts/train/collect_residents.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASSES = ["1_resident", "4_guest"]

# cam_01_9_d_20260707_114110_642457_msk_diff5.7_p1of1_conf0.87.jpg
_RE_FRAME = re.compile(
    r"^(?P<cam>cam_[^_]+)_\d+_[a-z]_(?P<date>\d{8})_(?P<time>\d{6})_\d+_.*_conf(?P<conf>\d+\.\d+)",
    re.IGNORECASE,
)
_RE_PERSONS = re.compile(r"_p\d+of(?P<total>\d+)_", re.IGNORECASE)


def _load_labels(src_dir: Path) -> dict[str, str]:
    """Загружает labels.json из родительского каталога (date_dir).
    Возвращает словарь {filename: class_label}."""
    lf = src_dir.parent / "labels.json"
    if not lf.exists():
        return {}
    try:
        with open(lf) as fh:
            data = json.load(fh)
        raw = data.get("labels", data)
        return {Path(k).name: v for k, v in raw.items() if isinstance(v, str)}
    except Exception:
        return {}


def _persons_in_frame(name: str) -> int:
    """Возвращает общее число людей в кадре из суффикса _pXofN_, или 1 если не распознано."""
    m = _RE_PERSONS.search(name)
    return int(m.group("total")) if m else 1


def _parse_frame(name: str) -> tuple[str, str, int, float] | None:
    """Возвращает (cam, date, seconds_of_day, conf) или None."""
    m = _RE_FRAME.match(name)
    if not m:
        return None
    t = m.group("time")  # HHMMSS
    sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    conf = float(m.group("conf"))
    return m.group("cam"), m.group("date"), sod, conf


def _default_sources(classes: list[str] = DEFAULT_CLASSES) -> list[Path]:
    groups = REPO_ROOT / ".data" / "groups"
    sources: list[Path] = []

    # датасет v1 (основная размеченная база)
    for cls in classes:
        d = groups / "v1" / "dataset" / cls
        if d.is_dir():
            sources.append(d)

    # инференс v1
    inference_root = groups / "v1" / "inference" / "images"
    if inference_root.is_dir():
        for date_dir in sorted(inference_root.iterdir()):
            for cls in classes:
                d = date_dir / cls
                if d.is_dir():
                    sources.append(d)

    return sources


def collect(
    sources: list[Path],
    out_dir: Path,
    *,
    dry_run: bool = False,
    ext: set[str] = IMAGE_EXTS,
    interval: int = 0,
    conf_delta: float = 0.0,
    max_persons: int = 0,
    collect_classes: set[str] | None = None,
) -> tuple[int, int, int, int]:
    """Копирует файлы из sources в out_dir.
    Возвращает (скопировано, пропущено_dedup, пропущено_persons, пропущено_labels).

    conf_delta > 0: из каждого окна берём все кадры с conf ≥ max(окна) − conf_delta.
    conf_delta = 0 (default): только один кадр с максимальным confidence.
    """
    if collect_classes is None:
        collect_classes = set(DEFAULT_CLASSES)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = {f.name for f in out_dir.iterdir() if f.is_file()}

    # Проход 1: собираем кандидатов (дедуп по имени + labels.json + фильтр по числу людей)
    candidates: list[Path] = []
    skipped_persons = 0
    skipped_labels = 0
    for src_dir in sources:
        if not src_dir.is_dir():
            print(f"  [!] Не найдено: {src_dir}", file=sys.stderr)
            continue
        labels = _load_labels(src_dir)
        for f in sorted(f for f in src_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in ext):
            if f.name in seen_names:
                continue
            seen_names.add(f.name)
            if f.name in labels and labels[f.name] not in collect_classes:
                skipped_labels += 1
                continue
            if max_persons > 0 and _persons_in_frame(f.stem) > max_persons:
                skipped_persons += 1
                continue
            candidates.append(f)

    # Проход 2: временна́я дедупликация
    if interval > 0:
        if conf_delta > 0:
            # Двухпроходный: сначала найти max(conf) в каждом окне,
            # затем взять все кадры с conf ≥ max − conf_delta.
            from collections import defaultdict as _dd
            window_max: dict[tuple, float] = {}
            window_frames: dict[tuple, list[tuple[float, Path]]] = _dd(list)
            noparse: list[Path] = []
            for f in candidates:
                parsed = _parse_frame(f.stem)
                if parsed is None:
                    noparse.append(f)
                    continue
                cam, date, sod, conf = parsed
                key = (cam, date, sod // interval)
                window_max[key] = max(window_max.get(key, -1.0), conf)
                window_frames[key].append((conf, f))
            to_copy: list[Path] = noparse[:]
            for key, items in window_frames.items():
                mx = window_max[key]
                to_copy.extend(f for c, f in items if c >= mx - conf_delta)
        else:
            # Только один кадр с максимальным confidence на окно.
            temporal: dict[tuple, tuple[float, Path]] = {}
            for f in candidates:
                parsed = _parse_frame(f.stem)
                if parsed is None:
                    key = ("__noparse__", f.name)
                    temporal[key] = (0.0, f)
                    continue
                cam, date, sod, conf = parsed
                key = (cam, date, sod // interval)
                prev_conf, _ = temporal.get(key, (-1.0, None))
                if conf > prev_conf:
                    temporal[key] = (conf, f)
            to_copy = [f for _, f in temporal.values()]
    else:
        to_copy = candidates

    skipped_dedup = len(candidates) - len(to_copy)

    copied = 0
    for f in sorted(to_copy, key=lambda p: p.name):
        if not dry_run:
            shutil.copy2(f, out_dir / f.name)
        copied += 1

    return copied, skipped_dedup, skipped_persons, skipped_labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Сбор кропов жителей для датасета Модели 2")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, metavar="CLS",
                    help=f"Классы для сбора (default: {' '.join(DEFAULT_CLASSES)})")
    ap.add_argument("--src", type=Path, action="append", dest="sources", metavar="DIR",
                    help="Каталог с кропами (можно передавать несколько раз). "
                         "По умолчанию: dataset/{classes}/ + inference/images/*/{classes}/")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v1" / "new",
                    help="Куда копировать (default: .data/residents/v1/new/)")
    ap.add_argument("--interval", type=int, default=2, metavar="SEC",
                    help="Временна́я дедупликация: один кадр на N секунд на камеру. "
                         "0 — отключить. (default: 2)")
    ap.add_argument("--conf-delta", type=float, default=0.0, metavar="F",
                    help="Из каждого окна брать все кадры с conf ≥ max(окна) − F. "
                         "0 — только максимальный (default).")
    ap.add_argument("--max-persons", type=int, default=1, metavar="N",
                    help="Отбрасывать кадры где в оригинале было больше N человек "
                         "(парсит _pXofN_). 0 — отключить. (default: 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Показать что будет скопировано, не копировать")
    args = ap.parse_args()

    sources = args.sources or _default_sources(args.classes)
    if not sources:
        print("[!] Источники не найдены.", file=sys.stderr)
        return 1

    print(f"Источники ({len(sources)}):")
    for s in sources:
        n = sum(1 for f in s.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS) if s.is_dir() else 0
        print(f"  {s}  [{n} файлов]")
    print(f"Классы:      {', '.join(args.classes)}")
    print(f"Назначение:  {args.out}")
    print(f"Интервал:    {args.interval}s" if args.interval > 0 else "Интервал:    отключён")
    print(f"conf-delta:  {args.conf_delta}" if args.conf_delta > 0 else "conf-delta:  только максимальный")
    print(f"Макс. людей: {args.max_persons}" if args.max_persons > 0 else "Макс. людей: без ограничений")
    if args.dry_run:
        print("(dry-run — файлы не копируются)")
    print()

    copied, skipped_dedup, skipped_persons, skipped_labels = collect(
        sources, args.out,
        dry_run=args.dry_run,
        interval=args.interval,
        conf_delta=args.conf_delta,
        max_persons=args.max_persons,
        collect_classes=set(args.classes),
    )

    if skipped_labels:
        print(f"Отброшено (labels.json → другой класс): {skipped_labels}")
    if args.max_persons > 0:
        print(f"Отброшено (кадры с >1 человеком): {skipped_persons}")
    if args.interval > 0:
        print(f"Отброшено (временна́я дедупликация): {skipped_dedup}")
    print(f"Скопировано: {copied}")
    total = sum(1 for f in args.out.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS) if args.out.is_dir() else 0
    print(f"Итого в {args.out}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
