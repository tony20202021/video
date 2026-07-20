"""
3b — темпоральное сглаживание классов Модели 1 (между classify и identify).

Читает выход 3_classify_groups (images/<date>/classifications.csv + раскладку single/uncertain),
сглаживает предсказания по времени (visit-HMM, src/common/utils/temporal_smooth) и приводит
раскладку + labels.json к СГЛАЖЕННЫМ классам — чтобы identify и downstream видели исправленные
классы. Скорость не важна (офлайн на сервере после копирования), важна точность.

Что делает:
  - по classifications.csv строит последовательности (камера из имени, время=ts_epoch, вероятности p_<class>);
  - сегментация по паузам во времени на визиты + Viterbi/HMM внутри визита (опц. гейтинг по уверенности);
  - если сглаженный класс ≠ текущей раскладки → перекладывает кроп в single/<class>/ и правит labels.json;
  - uncertain-кропы получают класс визита (спасаются); multi/unknown не трогаются;
  - пишет classifications_smoothed.csv (аудит: model_class → smoothed_class).
Идемпотентно: сглаживание считается от ИСХОДНЫХ вероятностей из CSV, а не от текущей раскладки.

Usage:
    python scripts/pipeline/3b_smooth_groups.py .data/groups/v4/inference/images        # все даты
    python scripts/pipeline/3b_smooth_groups.py .data/groups/v4/inference/images/20260720 --once
    python scripts/pipeline/3b_smooth_groups.py <root> --gap 60 --p-stay 0.95 --gate 0.8
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.classes import GROUP_CLASSES, RESIDENT_CLASS
from common.utils import multilabel as _ml
from common.utils import temporal_smooth as _ts
import logging
from common.utils.log_setup import setup_logging, add_file_handler

logger = logging.getLogger(__name__)
MSK = timezone(timedelta(hours=3))
GUEST_CLASS = "4_guest"
IDENTIFY_CLASSES = {RESIDENT_CLASS, GUEST_CLASS}
_DATE_RE = re.compile(r"^\d{8}$")


def _parse_cam_t(name: str) -> tuple[str, float] | None:
    """Камера + время (сек от начала суток) из имени кропа. None если не разобрать."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if (len(p) == 8 and p.isdigit() and i + 1 < len(parts)
                and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
            t = parts[i + 1]
            micro = parts[i + 2] if i + 2 < len(parts) and parts[i + 2].isdigit() else "0"
            sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + float(f"0.{micro}")
            return "_".join(parts[:i]), sod
    return None


def _index_files(date_dir: Path, ext: str = "jpg") -> dict[str, Path]:
    """{имя_файла: текущий_путь} по всему каталогу даты (кроме meta/)."""
    idx: dict[str, Path] = {}
    for f in date_dir.rglob(f"*.{ext}"):
        if "meta" in f.relative_to(date_dir).parts:
            continue
        idx[f.name] = f
    return idx


def _current_class(rel_parts: tuple[str, ...]) -> str:
    """Класс по расположению файла: single/<class>/ → class; uncertain/ → uncertain; multi/ → multi."""
    if len(rel_parts) >= 2 and rel_parts[0] == "single":
        return rel_parts[1]
    return rel_parts[0] if rel_parts else ""


def _read_csv_rows(date_dir: Path) -> list[dict]:
    csv_path = date_dir / "classifications.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def smooth_date(date_dir: Path, *, gap: float, p_stay: float, gate: float | None,
                include_uncertain: bool, ext: str = "jpg") -> dict:
    """Сглаживает один каталог-дату. Возвращает статистику."""
    rows = _read_csv_rows(date_dir)
    if not rows:
        return {"date": date_dir.name, "rows": 0}

    files = _index_files(date_dir, ext)
    # по имени кропа: исходные вероятности (первое вхождение — CSV накопительный, вероятности стабильны)
    recs: list[dict] = []
    meta: dict[str, dict] = {}
    for r in rows:
        name = r.get("crop", "")
        if not name or name in meta:
            continue
        oc = r.get("out_class", "")
        eligible = oc in GROUP_CLASSES or (include_uncertain and oc == "uncertain")
        if not eligible:
            continue
        try:
            probs = [float(r.get(f"p_{c}", 0) or 0) for c in GROUP_CLASSES]
        except ValueError:
            continue
        pr = _parse_cam_t(name)
        if pr is None:
            continue
        cam, t = pr
        # предпочитаем абсолютный ts_epoch, если есть (корректно через полночь)
        try:
            te = r.get("ts_epoch", "")
            if te:
                t = float(te)
        except ValueError:
            pass
        rec = {"cam": cam, "t": t, "probs": probs, "name": name, "model_class": r.get("group", oc)}
        meta[name] = rec
        recs.append(rec)

    if not recs:
        return {"date": date_dir.name, "rows": len(rows), "eligible": 0}

    smoothed = _ts.smooth_sequence(recs, GROUP_CLASSES, gap_sec=gap, p_stay=p_stay, gate=gate)

    labels = _ml.load_labels(date_dir / "labels.json")   # {rel: [classes]}
    moved = changed = rescued = 0
    audit = []
    for rec, (sm_class, _changed) in zip(recs, smoothed):
        name = rec["name"]
        f = files.get(name)
        if f is None or not f.is_file():
            continue
        rel_parts = f.relative_to(date_dir).parts
        cur = _current_class(rel_parts)     # текущая раскладка (что было до сглаживания)
        audit.append({"crop": name, "cam": rec["cam"], "model_class": cur,
                      "smoothed_class": sm_class, "changed": cur != sm_class})
        if cur == sm_class:
            continue
        # перекладываем в single/<sm_class>/ + правим labels.json
        dst_dir = date_dir / "single" / sm_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        old_rel = f.relative_to(date_dir).as_posix()
        if f.resolve() != dst.resolve() and not dst.exists():
            shutil.move(str(f), str(dst))
            moved += 1
        labels.pop(old_rel, None)
        labels[(Path("single") / sm_class / name).as_posix()] = [sm_class]
        changed += 1
        if cur == "uncertain":
            rescued += 1

    _ml.save_labels(date_dir / "labels.json", labels, task="classify")

    # аудит-CSV
    with open(date_dir / "classifications_smoothed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "cam", "model_class", "smoothed_class", "changed"])
        w.writeheader()
        w.writerows(audit)

    n_ident = sum(1 for a in audit if a["smoothed_class"] in IDENTIFY_CLASSES)
    return {"date": date_dir.name, "rows": len(rows), "eligible": len(recs),
            "changed": changed, "moved": moved, "rescued_uncertain": rescued,
            "to_identify": n_ident}


def _date_dirs(root: Path) -> list[Path]:
    """Каталоги-даты YYYYMMDD внутри root; если root сам дата — [root]."""
    if _DATE_RE.match(root.name):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _DATE_RE.match(d.name))


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description="Темпоральное сглаживание классов Модели 1 (3b)")
    ap.add_argument("input_dir", type=Path,
                    help="images/ (все даты) или конкретный каталог-дата")
    ap.add_argument("--gap", type=float, default=_ts.DEFAULT_GAP_SEC,
                    help=f"пауза (сек) для сегментации визитов (default: {_ts.DEFAULT_GAP_SEC})")
    ap.add_argument("--p-stay", type=float, default=_ts.DEFAULT_P_STAY,
                    help=f"вероятность сохранения класса в HMM (default: {_ts.DEFAULT_P_STAY})")
    ap.add_argument("--gate", type=float, default=None,
                    help="не трогать предсказания с уверенностью ≥ gate (default: выкл)")
    ap.add_argument("--no-uncertain", dest="include_uncertain", action="store_false",
                    help="не спасать uncertain-кропы (по умолчанию спасаем — даём класс визита)")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=120.0)
    args = ap.parse_args()

    if not args.input_dir.exists():
        logger.error("Не найдено: %s", args.input_dir)
        return 1

    logger.info("3b smooth: gap=%.0fс p_stay=%.2f gate=%s uncertain=%s",
                args.gap, args.p_stay, args.gate, args.include_uncertain)

    while True:
        dates = _date_dirs(args.input_dir)
        total_changed = 0
        for dd in dates:
            st = smooth_date(dd, gap=args.gap, p_stay=args.p_stay, gate=args.gate,
                             include_uncertain=args.include_uncertain, ext=args.ext)
            if st.get("changed"):
                logger.info("1 батч (%d кадров)  Готово.  %s: сглажено %d (переложено %d, "
                            "uncertain→класс %d, в identify %d)",
                            st.get("eligible", 0), st["date"], st["changed"], st["moved"],
                            st["rescued_uncertain"], st["to_identify"])
                total_changed += st["changed"]
        if total_changed == 0:
            logger.info("(3b_smooth_groups) Изменений нет — ожидание %.0fs…", args.poll_sec)
        if args.once:
            break
        time.sleep(args.poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
