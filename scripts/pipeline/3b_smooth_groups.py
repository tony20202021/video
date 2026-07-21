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


_RE_PERSON = re.compile(r"p\d+of(\d+)")


def _persons_in_frame(name: str) -> int:
    """Число людей в кадре из 'pNofM' (M). 1 если не найдено."""
    m = _RE_PERSON.search(name)
    return int(m.group(1)) if m else 1


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


# ─── Визуальный дебаг сглаживания (--viz) ─────────────────────────────────────
# Для каждого исправленного кропа рисуем конкат кадров ВИЗИТА (на основе которых
# Viterbi принял решение): миниатюры по времени, исправленный кадр в рамке, под
# каждым — время/дата, сверху — вероятности модели и новый класс. Латиница в
# подписях (cv2 не рисует кириллицу). Ошибки виз-рендера не роняют пайплайн.
from common.utils.classes import GROUP_CLASS_COLORS

_SHORT = {"1_resident": "res", "2_delivery": "del", "3_utilities": "utl", "4_guest": "gst"}


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))   # BGR


_CLASS_BGR = {c: _hex_bgr(GROUP_CLASS_COLORS.get(c, "#888888")) for c in GROUP_CLASSES}


def _fmt_dt(t: float, date_str: str) -> tuple[str, str]:
    """(ЧЧ:ММ:СС, ГГГГ-ММ-ДД) из ts_epoch или секунд-от-начала-суток."""
    if t and t > 1_000_000:                      # абсолютный epoch
        dt = datetime.fromtimestamp(t, MSK)
        return dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d")
    ti = int(t or 0)                             # секунды от начала суток
    ds = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
    return f"{ti // 3600:02d}:{(ti % 3600) // 60:02d}:{ti % 60:02d}", ds


def _visits_by_rec(recs: list[dict], gap: float) -> dict[int, list[int]]:
    """rec-индекс → список rec-индексов его визита (та же группировка, что в smooth_sequence)."""
    by_cam: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        by_cam.setdefault(r.get("cam", ""), []).append(i)
    out: dict[int, list[int]] = {}
    for _cam, ids in by_cam.items():
        ids.sort(key=lambda i: recs[i]["t"])
        tms = [recs[i]["t"] for i in ids]
        for vis in _ts.segment_visits(tms, gap):
            members = [ids[j] for j in vis]
            for i in members:
                out[i] = members
    return out


def _render_smoothing_viz(center_i: int, visit: list[int], recs: list[dict],
                          files: dict, date_str: str, sm_class: str,
                          out_path: Path, *, window: int = 10) -> bool:
    """Рисует конкат визита с выделенным исправленным кадром. True если сохранил."""
    import cv2
    import numpy as np
    if center_i not in visit:
        return False
    cpos = visit.index(center_i)
    lo, hi = max(0, cpos - window), min(len(visit), cpos + window + 1)
    seg = visit[lo:hi]

    TH, MAXW, LAB = 150, 150, 32
    tiles = []
    for j in seg:
        rec = recs[j]
        f = files.get(rec["name"])
        img = cv2.imread(str(f)) if f and Path(f).is_file() else None
        if img is None:
            img = np.full((TH, MAXW, 3), 40, np.uint8)
        else:
            h, w = img.shape[:2]
            nw = min(MAXW, max(1, int(w * (TH / h))))
            img = cv2.resize(img, (nw, TH))
        tw = img.shape[1]
        canvas = np.full((TH + LAB, tw, 3), 25, np.uint8)
        canvas[:TH, :tw] = img
        hhmmss, _ymd = _fmt_dt(rec["t"], date_str)
        m_idx = int(np.argmax(rec["probs"]))
        cv2.putText(canvas, hhmmss, (2, TH + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, _SHORT.get(GROUP_CLASSES[m_idx], "?"), (2, TH + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _CLASS_BGR[GROUP_CLASSES[m_idx]], 1, cv2.LINE_AA)
        if j == center_i:                        # исправленный кадр — жёлтая рамка
            cv2.rectangle(canvas, (0, 0), (tw - 1, TH - 1), (0, 215, 255), 3)
        tiles.append(canvas)

    # выравнивание по ширине (hconcat требует равной высоты — она уже TH+LAB)
    strip = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
    W = strip.shape[1]

    # заголовок: имя + вероятности модели центра + новый класс
    rec = recs[center_i]
    probs = rec["probs"]
    hhmmss, ymd = _fmt_dt(rec["t"], date_str)
    prob_txt = "  ".join(f"{_SHORT.get(c, c)} {probs[k]:.2f}" for k, c in enumerate(GROUP_CLASSES))
    m_idx = int(np.argmax(probs))
    HH = 60
    header = np.full((HH, W, 3), 15, np.uint8)
    cv2.putText(header, f"{rec['name']}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(header, f"model: {prob_txt}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, f"{ymd} {hhmmss}   {_SHORT.get(GROUP_CLASSES[m_idx],'?')} -> ",
                (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    (tw_, _), _ = cv2.getTextSize(f"{ymd} {hhmmss}   {_SHORT.get(GROUP_CLASSES[m_idx],'?')} -> ",
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(header, _SHORT.get(sm_class, sm_class), (6 + tw_, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CLASS_BGR.get(sm_class, (0, 215, 255)), 2, cv2.LINE_AA)

    full = cv2.vconcat([header, strip])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full)
    return True


def smooth_date(date_dir: Path, *, gap: float, p_stay: float, gate: float | None,
                include_uncertain: bool, max_persons: int = 1, ext: str = "jpg",
                viz: bool = False, viz_dir: Path | None = None) -> dict:
    """Сглаживает один каталог-дату. Возвращает статистику.
    max_persons: сглаживаем только кропы из кадров с ≤ max_persons людей (pNofM) — многолюдные
    кадры содержат РАЗНЫХ людей, их сглаживать по времени нельзя (см. замер: ломает гостей).
    viz: для каждого исправленного кропа рисует конкат визита в viz_dir (дебаг сглаживания)."""
    rows = _read_csv_rows(date_dir)
    if not rows:
        return {"date": date_dir.name, "rows": 0}

    files = _index_files(date_dir, ext)
    # по имени кропа: исходные вероятности (первое вхождение — CSV накопительный, вероятности стабильны)
    recs: list[dict] = []
    meta: dict[str, dict] = {}
    skipped_crowd = 0
    for r in rows:
        name = r.get("crop", "")
        if not name or name in meta:
            continue
        oc = r.get("out_class", "")
        eligible = oc in GROUP_CLASSES or (include_uncertain and oc == "uncertain")
        if not eligible:
            continue
        if _persons_in_frame(name) > max_persons:   # многолюдный кадр — не сглаживаем
            skipped_crowd += 1
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
    moved = changed = rescued = viz_n = 0
    audit = []
    corrections = []   # (rec_idx, name, path, cur_class, smoothed_class)
    for i, (rec, (sm_class, _changed)) in enumerate(zip(recs, smoothed)):
        name = rec["name"]
        f = files.get(name)
        if f is None or not f.is_file():
            continue
        rel_parts = f.relative_to(date_dir).parts
        cur = _current_class(rel_parts)     # текущая раскладка (что было до сглаживания)
        audit.append({"crop": name, "cam": rec["cam"], "model_class": cur,
                      "smoothed_class": sm_class, "changed": cur != sm_class})
        if cur != sm_class:
            corrections.append((i, name, f, cur, sm_class))

    # виз-дебаг: рисуем ДО перемещений — все кадры визитов ещё на исходных местах
    if viz and corrections:
        vdir = viz_dir or (date_dir / "meta" / "smooth_viz")
        if vdir.is_dir():
            for _old in vdir.glob("*.jpg"):
                _old.unlink()
        visit_of = _visits_by_rec(recs, gap)
        for i, name, _f, _cur, sm_class in corrections:
            try:
                if _render_smoothing_viz(i, visit_of.get(i, [i]), recs, files,
                                         date_dir.name, sm_class,
                                         vdir / f"{Path(name).stem}.jpg"):
                    viz_n += 1
            except Exception as e:                       # виз не должен ронять пайплайн
                logger.warning("viz fail %s: %s", name, e)

    # перекладываем исправленные кропы в single/<sm_class>/ + правим labels.json
    for i, name, f, cur, sm_class in corrections:
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
            "skipped_crowd": skipped_crowd, "to_identify": n_ident, "viz": viz_n}


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
    ap.add_argument("--gate", type=float, default=0.8,
                    help="не трогать предсказания с уверенностью ≥ gate (default: 0.8; 0=выкл)")
    ap.add_argument("--max-persons", type=int, default=1,
                    help="сглаживать только кадры с ≤ N людей (default: 1 — многолюдные не трогаем)")
    ap.add_argument("--no-uncertain", dest="include_uncertain", action="store_false",
                    help="не спасать uncertain-кропы (по умолчанию спасаем — даём класс визита)")
    ap.add_argument("--viz", action="store_true",
                    help="дебаг: для каждого исправленного кропа рисовать конкат визита "
                         "в <date>/meta/smooth_viz/ (по умолчанию выкл)")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=120.0)
    args = ap.parse_args()

    if not args.input_dir.exists():
        logger.error("Не найдено: %s", args.input_dir)
        return 1

    gate = args.gate if args.gate and args.gate > 0 else None
    logger.info("3b smooth: gap=%.0fс p_stay=%.2f gate=%s max_persons=%d uncertain=%s viz=%s",
                args.gap, args.p_stay, gate, args.max_persons, args.include_uncertain, args.viz)

    while True:
        dates = _date_dirs(args.input_dir)
        total_changed = 0
        for dd in dates:
            st = smooth_date(dd, gap=args.gap, p_stay=args.p_stay, gate=gate,
                             include_uncertain=args.include_uncertain,
                             max_persons=args.max_persons, ext=args.ext, viz=args.viz)
            if st.get("changed"):
                logger.info("1 батч (%d кадров)  Готово.  %s: сглажено %d (переложено %d, "
                            "uncertain→класс %d, в identify %d, viz %d)",
                            st.get("eligible", 0), st["date"], st["changed"], st["moved"],
                            st["rescued_uncertain"], st["to_identify"], st.get("viz", 0))
                total_changed += st["changed"]
        if total_changed == 0:
            logger.info("(3b_smooth_groups) Изменений нет — ожидание %.0fs…", args.poll_sec)
        if args.once:
            break
        time.sleep(args.poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
