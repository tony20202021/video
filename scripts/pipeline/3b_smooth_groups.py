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
import json
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


def _write_watermark(date_dir: Path, *, csv_rows: int, eligible: int, changed: int) -> None:
    """Пишет <date>/meta/smooth_state.json — «сглажено до csv_rows строк CSV».
    identify (4) читает его и не берёт дату, пока smooth не догнал текущий classifications.csv
    (иначе успевает опознать ложного резидента до перекладки — см. docs/services.md)."""
    meta = date_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "smooth_state.json").write_text(
        json.dumps({"csv_rows": csv_rows, "eligible": eligible, "changed": changed,
                    "smoother": "3b"}, ensure_ascii=False),
        encoding="utf-8")


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


def _frames_by_cam(rows: list[dict]) -> dict[str, list[dict]]:
    """ВСЕ кадры камеры (вкл. M>1 и multi/uncertain) по времени — для viz-контекста.
    [{name, t, probs, M, out}]. Дедуп по имени (CSV накопительный)."""
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for r in rows:
        nm = r.get("crop", "")
        if not nm or nm in seen:
            continue
        pr = _parse_cam_t(nm)
        if pr is None:
            continue
        seen.add(nm)
        cam, t = pr
        te = r.get("ts_epoch", "")
        try:
            if te:
                t = float(te)
        except ValueError:
            pass
        try:
            probs = [float(r.get(f"p_{c}", 0) or 0) for c in GROUP_CLASSES]
        except ValueError:
            probs = [0.0] * len(GROUP_CLASSES)
        out.setdefault(cam, []).append({"name": nm, "t": t, "probs": probs,
                                        "M": _persons_in_frame(nm), "out": r.get("out_class", "")})
    for cam in out:
        out[cam].sort(key=lambda x: x["t"])
    return out


_PROB_DX = 26   # шаг между колонками вероятностей в подписи тайла


def _draw_probs(canvas, x: int, y: int, probs: list, *, labels: bool = False) -> None:
    """4 числа-процента вероятностей, каждое в цвете класса (r/d/u/g). labels: строкой ниже —
    буквенные названия классов (res/del/utl/gst) под соответствующими числами."""
    import cv2
    cx = x
    for k, c in enumerate(GROUP_CLASSES):
        cv2.putText(canvas, f"{int(round(probs[k] * 100)):02d}", (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _CLASS_BGR[c], 1, cv2.LINE_AA)
        if labels:
            cv2.putText(canvas, _SHORT.get(c, c), (cx, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _CLASS_BGR[c], 1, cv2.LINE_AA)
        cx += _PROB_DX


def _draw_prob_graph(width: int, pts: list, height: int = 96):
    """График вероятностей по времени на всю ширину конката: для каждого кадра — 4 точки
    (по классам, высота=вероятность), точки одного класса соединены отрезками. pts: [(x_center, probs)]."""
    import cv2
    import numpy as np
    g = np.full((height, width, 3), 18, np.uint8)
    top, bot, left = 9, height - 10, 22
    for val, lab in ((1.0, "1"), (0.5, ".5"), (0.0, "0")):        # сетка 0/.5/1
        y = int(top + (1 - val) * (bot - top))
        cv2.line(g, (left, y), (width - 3, y), (45, 45, 55), 1)
        cv2.putText(g, lab, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1, cv2.LINE_AA)
    for k, c in enumerate(GROUP_CLASSES):
        col = _CLASS_BGR[c]
        poly = [(int(x), int(top + (1 - pr[k]) * (bot - top))) for x, pr in pts]
        if len(poly) >= 2:
            cv2.polylines(g, [np.array(poly, np.int32)], False, col, 1, cv2.LINE_AA)
        for (x, y) in poly:
            cv2.circle(g, (x, y), 2, col, -1, cv2.LINE_AA)
    return g


def _render_smoothing_viz(center_name: str, cam_frames: list[dict], sm_class: str,
                          files: dict, date_str: str, out_path: Path,
                          *, win_sec: float = 90.0, max_tiles: int = 41) -> bool:
    """Конкат кадров камеры в окне ±win_sec СЕКУНД вокруг исправленного (по ВСЕМ кадрам, вкл. M>1).
    Окно по ВРЕМЕНИ (не по числу кадров) — иначе на редкой камере попадут события со всего дня.
    Под каждым кадром: время, класс-argmax модели, вероятности (r/d/u/g %). Исправленный —
    жёлтая рамка + argmax→новый класс рядом. M>1 — красная рамка (не сглаживается)."""
    import cv2
    import numpy as np
    pos = next((k for k, fr in enumerate(cam_frames) if fr["name"] == center_name), None)
    if pos is None:
        return False
    ct = cam_frames[pos]["t"]
    near = [fr for fr in cam_frames if abs(fr["t"] - ct) <= win_sec]   # окно по времени
    ci = next(k for k, fr in enumerate(near) if fr["name"] == center_name)
    if len(near) > max_tiles:                                          # ограничение ширины
        half = max_tiles // 2
        near = near[max(0, ci - half): ci + half + 1]
    seg = near

    TH, MAXW, LAB = 150, 150, 66
    tiles, pts, xoff = [], [], 0
    for fr in seg:
        is_center = fr["name"] == center_name
        M = fr["M"]
        f = files.get(fr["name"])
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
        hhmmss, _ymd = _fmt_dt(fr["t"], date_str)
        m_idx = int(np.argmax(fr["probs"]))
        cv2.putText(canvas, hhmmss, (2, TH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (210, 210, 210), 1, cv2.LINE_AA)
        if M > 1:                                # многолюдный кадр — не сглаживается
            cv2.putText(canvas, f"M{M}", (tw - 30, TH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (60, 60, 235), 1, cv2.LINE_AA)
        if is_center:                            # argmax → новый класс прямо у кадра
            base = f"{_SHORT.get(GROUP_CLASSES[m_idx], '?')}->"
            cv2.putText(canvas, base, (2, TH + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (150, 150, 150), 1, cv2.LINE_AA)
            (bw, _), _ = cv2.getTextSize(base, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(canvas, _SHORT.get(sm_class, sm_class), (2 + bw, TH + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CLASS_BGR.get(sm_class, (0, 215, 255)), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, _SHORT.get(GROUP_CLASSES[m_idx], "?"), (2, TH + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, _CLASS_BGR[GROUP_CLASSES[m_idx]], 1, cv2.LINE_AA)
        _draw_probs(canvas, 2, TH + 50, fr["probs"], labels=True)   # числа + буквы классов ниже
        if M > 1:
            cv2.rectangle(canvas, (0, 0), (tw - 1, TH - 1), (60, 60, 200), 2)   # красная = M>1
        if is_center:
            cv2.rectangle(canvas, (0, 0), (tw - 1, TH - 1), (0, 215, 255), 3)   # жёлтая = исправлен
        tiles.append(canvas)
        pts.append((xoff + tw / 2.0, fr["probs"]))                 # точка графика — центр тайла
        xoff += tw

    strip = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
    graph = _draw_prob_graph(strip.shape[1], pts)                  # график probs по времени на всю ширину

    HH = 40
    legend = "yellow=fixed  red=M>1(not smoothed)  nums/graph=probs r/d/u/g %"
    # ширина = max(полоса, самая длинная строка заголовка) — иначе на узком конкате текст обрезается
    (nw_, _), _ = cv2.getTextSize(center_name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    (lw_, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    W = max(strip.shape[1], nw_ + 12, lw_ + 12)
    header = np.full((HH, W, 3), 15, np.uint8)
    cv2.putText(header, center_name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(header, legend, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 120), 1, cv2.LINE_AA)
    if W > strip.shape[1]:                        # добить полосу и график справа фоном до ширины заголовка
        strip = cv2.hconcat([strip, np.full((strip.shape[0], W - strip.shape[1], 3), 25, np.uint8)])
        graph = cv2.hconcat([graph, np.full((graph.shape[0], W - graph.shape[1], 3), 18, np.uint8)])

    full = cv2.vconcat([header, strip, graph])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full)
    return True


def smooth_date(date_dir: Path, *, gap: float, p_stay: float, gate: float | None,
                include_uncertain: bool, max_persons: int = 1, ext: str = "jpg",
                viz: bool = False, viz_dir: Path | None = None,
                guard_sec: float | None = None, min_minority_prob: float | None = None,
                veto_prob: float | None = None) -> dict:
    """Сглаживает один каталог-дату. Возвращает статистику.
    max_persons: сглаживаем только кропы из кадров с ≤ max_persons людей (pNofM) — многолюдные
    кадры содержат РАЗНЫХ людей, их сглаживать по времени нельзя (см. замер: ломает гостей).
    guard_sec/min_minority_prob/veto_prob: предохранители против ложных флипов в резидента
    (temporal_smooth.smooth_sequence). ПО УМОЛЧАНИЮ ВЫКЛ (None): замер на 1101 ручной метке
    20260720 показал, что veto/min_minority меняют ~7 резидентов ради ~1-2 меньшинств — net хуже
    (8.0%→8.5-8.9%). Модель путает res/del/guest, «сильное меньшинство» часто = неуверенный
    резидент. Оставлены параметром для будущих замеров. viz: конкат визита в viz_dir (дебаг)."""
    rows = _read_csv_rows(date_dir)
    if not rows:
        return {"date": date_dir.name, "rows": 0}

    files = _index_files(date_dir, ext)
    # по имени кропа: исходные вероятности (первое вхождение — CSV накопительный, вероятности стабильны)
    recs: list[dict] = []
    context: list[dict] = []     # M>1 кадры — не сглаживаем, но держим как контекст (вето по гостям)
    meta: dict[str, dict] = {}
    seen_ctx: set[str] = set()
    skipped_crowd = 0
    for r in rows:
        name = r.get("crop", "")
        if not name:
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
        try:
            probs = [float(r.get(f"p_{c}", 0) or 0) for c in GROUP_CLASSES]
        except ValueError:
            continue
        if _persons_in_frame(name) > max_persons:   # многолюдный кадр — в контекст, не в сглаживание
            if name not in seen_ctx:
                seen_ctx.add(name)
                context.append({"cam": cam, "t": t, "probs": probs})
                skipped_crowd += 1
            continue
        if name in meta:
            continue
        oc = r.get("out_class", "")
        if not (oc in GROUP_CLASSES or (include_uncertain and oc == "uncertain")):
            continue
        rec = {"cam": cam, "t": t, "probs": probs, "name": name, "model_class": r.get("group", oc)}
        meta[name] = rec
        recs.append(rec)

    if not recs:
        _write_watermark(date_dir, csv_rows=len(rows), eligible=0, changed=0)
        return {"date": date_dir.name, "rows": len(rows), "eligible": 0}

    smoothed = _ts.smooth_sequence(
        recs, GROUP_CLASSES, gap_sec=gap, p_stay=p_stay, gate=gate,
        guard_sec=guard_sec, protect_class=RESIDENT_CLASS,
        minority_classes=[c for c in GROUP_CLASSES if c != RESIDENT_CLASS],
        min_minority_prob=min_minority_prob, context=context, veto_prob=veto_prob)

    labels = _ml.load_labels(date_dir / "labels.json")   # {rel: [classes]}
    moved = changed = rescued = viz_n = 0
    audit = []
    corrections = []   # (rec_idx, name, path, cur_class, smoothed_class) — перемещение раскладки
    viz_targets = []   # (rec_idx, name, smoothed_class) — где smoothing изменил класс vs МОДЕЛИ
    for i, (rec, (sm_class, changed)) in enumerate(zip(recs, smoothed)):
        name = rec["name"]
        f = files.get(name)
        if f is None or not f.is_file():
            continue
        rel_parts = f.relative_to(date_dir).parts
        cur = _current_class(rel_parts)     # текущая раскладка (что было до сглаживания)
        audit.append({"crop": name, "cam": rec["cam"], "model_class": cur,
                      "smoothed_class": sm_class, "changed": cur != sm_class})
        if changed:                         # sm != argmax модели → «исправление» (для viz, стабильно)
            viz_targets.append((i, name, sm_class))
        if cur != sm_class:                 # раскладка отличается от sm → переложить файл
            corrections.append((i, name, f, cur, sm_class))

    # виз-дебаг: показываем ВСЕ исправления модели (sm≠argmax) — стабильно между прогонами,
    # т.к. считается от вероятностей CSV, а не от раскладки (иначе после перекладки viz пропадал).
    # Перерисовываем только если CSV вырос с прошлой генерации (маркер .viz_rows) — не каждый поллинг.
    if viz and viz_targets:
        vdir = viz_dir or (date_dir / "meta" / "smooth_viz")
        marker = vdir / ".viz_rows"
        prev = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if prev == str(len(rows)) and any(vdir.glob("*.jpg")):
            pass                                         # CSV не менялся — viz актуален, не трогаем
        else:
            vdir.mkdir(parents=True, exist_ok=True)
            for _old in vdir.glob("*.jpg"):
                _old.unlink()
            cam_frames = _frames_by_cam(rows)            # ВСЕ кадры (вкл. M>1) для контекста
            name_to_cam = {r["name"]: r["cam"] for r in recs}
            for i, name, sm_class in viz_targets:
                try:
                    frames = cam_frames.get(name_to_cam.get(name, ""), [])
                    if _render_smoothing_viz(name, frames, sm_class, files,
                                             date_dir.name, vdir / f"{Path(name).stem}.jpg"):
                        viz_n += 1
                except Exception as e:                   # виз не должен ронять пайплайн
                    logger.warning("viz fail %s: %s", name, e)
            marker.write_text(str(len(rows)), encoding="utf-8")

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

    _write_watermark(date_dir, csv_rows=len(rows), eligible=len(recs), changed=changed)
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
