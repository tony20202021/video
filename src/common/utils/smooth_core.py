"""Общее ядро темпорального сглаживания — для групп (Модель 1) и жителей (Модель 2).

Читает <inference>/images/<date>/<csv_name> (classifications.csv или identifications.csv),
сглаживает по времени (Viterbi/HMM или бегущее окно, temporal_smooth), пишет ТОЛЬКО сайдкар
<inference>/smoothed/<date>/ (classifications_smoothed.csv crop→smoothed_class, smooth_state.json,
smooth_viz/). images/ НЕ мутируется. Набор классов, подписи и цвета берутся из конфига (SmoothCfg)
или выводятся из p_* колонок CSV (открытый набор жителей).

Драйверы-обёртки: scripts/pipeline/3b_smooth_groups.py (группы), 4b_smooth_identity.py (жители).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common.utils import multilabel as _ml
from common.utils import temporal_smooth as _ts
from common.utils.log_setup import setup_logging, add_file_handler

logger = logging.getLogger(__name__)
MSK = timezone(timedelta(hours=3))
_DATE_RE = re.compile(r"^\d{8}$")
_RE_PERSON = re.compile(r"p\d+of(\d+)")

# палитра для авто-цветов классов (открытый набор жителей) — раздаётся по индексу
_PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
            "#f032e6", "#bcf60c", "#fa8072", "#008080", "#e6beff", "#9a6324", "#fffac8",
            "#800000", "#aaffc3", "#808000", "#ffd8b1", "#4682b4", "#c0c0c0"]


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))   # BGR


def _auto_short(cls: str) -> str:
    """Компактная подпись класса для viz: инициалы слов + все цифры (уникально, ≤8)."""
    toks = cls.split("_")
    ini = "".join(t[0] for t in toks if t and not t[0].isdigit())
    nums = "".join(t for t in toks if t.isdigit())
    return (ini + nums)[:8] or cls[:8]


@dataclass
class SmoothCfg:
    """Статическая конфигурация одного сервиса сглаживания."""
    smoother: str                              # метка ("groups"/"identity") — в watermark и логах
    csv_name: str = "classifications.csv"      # имя входного CSV в images/<date>/
    fixed_classes: list | None = None          # фикс. набор (группы); None → вывести из p_* колонок
    resident_class: str | None = None          # protect_class для предохранителей (off по умолчанию)
    downstream_classes: set | None = None       # классы «на следующий шаг» (для счётчика to_next)
    fixed_short: dict | None = None            # подписи классов (иначе авто)
    fixed_colors: dict | None = None           # hex-цвета классов (иначе палитра по индексу)
    viz_topk: int = 4                          # сколько вероятностей рисовать под тайлом


@dataclass
class _Ctx:
    """Разрешённый под конкретную дату контекст: классы + стили."""
    classes: list
    idx: dict            # class → index
    short: dict          # class → короткая подпись
    colors: dict         # class → BGR
    resident_class: str | None
    downstream_classes: set
    topk: int


def _resolve_ctx(cfg: SmoothCfg, header: list) -> _Ctx:
    classes = list(cfg.fixed_classes) if cfg.fixed_classes else [
        c[2:] for c in header if c.startswith("p_")]
    short = dict(cfg.fixed_short) if cfg.fixed_short else {}
    colors = {}
    for i, c in enumerate(classes):
        short.setdefault(c, _auto_short(c))
        hexc = (cfg.fixed_colors or {}).get(c) or _PALETTE[i % len(_PALETTE)]
        colors[c] = _hex_bgr(hexc)
    return _Ctx(classes, {c: i for i, c in enumerate(classes)}, short, colors,
                cfg.resident_class, cfg.downstream_classes or set(), min(cfg.viz_topk, len(classes)))


# ─── общие парс-хелперы ───────────────────────────────────────────────────────
def _persons_in_frame(name: str) -> int:
    m = _RE_PERSON.search(name)
    return int(m.group(1)) if m else 1


def _parse_cam_t(name: str, merge_zones: bool = False) -> tuple[str, float] | None:
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if (len(p) == 8 and p.isdigit() and i + 1 < len(parts)
                and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
            t = parts[i + 1]
            micro = parts[i + 2] if i + 2 < len(parts) and parts[i + 2].isdigit() else "0"
            sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + float(f"0.{micro}")
            cam_parts = parts[:i]
            # merge_zones: снять суффикс ЗОНЫ (одиночная буква d/u) → зоны одной камеры в один поток.
            # Разные физические камеры (cam_01 vs cam_02) остаются раздельными — у них разный префикс.
            if merge_zones and cam_parts and len(cam_parts[-1]) == 1 and cam_parts[-1].isalpha():
                cam_parts = cam_parts[:-1]
            return "_".join(cam_parts), sod
    return None


def _split_zone(stem: str):
    """(prefix_parts, zone, suffix_parts): zone = одиночная буква (d/u) ПЕРЕД 8-значной датой.
    Если суффикса зоны нет — zone=None. Для сборки имени merged-конката с общей зоной (u+d)."""
    parts = stem.split("_")
    for i in range(1, len(parts)):
        if (len(parts[i]) == 8 and parts[i].isdigit()
                and i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
            if len(parts[i - 1]) == 1 and parts[i - 1].isalpha():
                return parts[:i - 1], parts[i - 1], parts[i:]
            return parts[:i], None, parts[i:]
    return None, None, None


def _index_files(date_dir: Path, ext: str = "jpg") -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for f in date_dir.rglob(f"*.{ext}"):
        if "meta" in f.relative_to(date_dir).parts:
            continue
        idx[f.name] = f
    return idx


def _current_class(rel_parts: tuple) -> str:
    """Класс по расположению файла: single/<class>/ → class; иначе первый компонент (<person_id>/uncertain/...)."""
    if len(rel_parts) >= 2 and rel_parts[0] == "single":
        return rel_parts[1]
    return rel_parts[0] if rel_parts else ""


def _read_csv_rows(date_dir: Path, csv_name: str) -> list[dict]:
    csv_path = date_dir / csv_name
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_watermark(out_dir: Path, *, csv_rows: int, eligible: int, changed: int,
                     sig: str = "", smoother: str = "smooth") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smooth_state.json").write_text(
        json.dumps({"csv_rows": csv_rows, "eligible": eligible, "changed": changed,
                    "smoother": smoother, "sig": sig}, ensure_ascii=False), encoding="utf-8")


def _fmt_dt(t: float, date_str: str) -> tuple[str, str]:
    if t and t > 1_000_000:
        dt = datetime.fromtimestamp(t, MSK)
        return dt.strftime("%H:%M:%S"), dt.strftime("%Y-%m-%d")
    ti = int(t or 0)
    ds = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
    return f"{ti // 3600:02d}:{(ti % 3600) // 60:02d}:{ti % 60:02d}", ds


def _frames_by_cam(rows: list[dict], ctx: _Ctx, merge_zones: bool = False) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for r in rows:
        nm = r.get("crop", "")
        if not nm or nm in seen:
            continue
        pr = _parse_cam_t(nm, merge_zones)
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
            probs = [float(r.get(f"p_{c}", 0) or 0) for c in ctx.classes]
        except ValueError:
            probs = [0.0] * len(ctx.classes)
        out.setdefault(cam, []).append({"name": nm, "t": t, "probs": probs,
                                        "M": _persons_in_frame(nm)})
    for cam in out:
        out[cam].sort(key=lambda x: x["t"])
    return out


# ─── viz (обобщён под N классов) ──────────────────────────────────────────────
def _draw_probs(canvas, x: int, y: int, probs: list, ctx: _Ctx, *, labels: bool = False) -> None:
    """top-K вероятностей (по убыванию) — число% + короткая подпись класса, в цвете класса.
    Для групп topk=4 (все), для жителей topk≈3 (иначе N чисел не влезают)."""
    import cv2
    order = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)[:ctx.topk]
    cx, dx = x, 44
    for k in order:
        c = ctx.classes[k]
        cv2.putText(canvas, f"{int(round(probs[k] * 100)):02d}", (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, ctx.colors[c], 1, cv2.LINE_AA)
        if labels:
            cv2.putText(canvas, ctx.short[c][:6], (cx, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, ctx.colors[c], 1, cv2.LINE_AA)
        cx += dx


def _draw_prob_graph(width: int, pts: list, ctx: _Ctx, plot_idx: list, height: int = 96, title: str = ""):
    """График вероятностей по времени. Рисуем ТОЛЬКО классы plot_idx (те, что где-то argmax) —
    иначе на N жителях каша. pts: [(x_center, probs)]."""
    import cv2
    import numpy as np
    g = np.full((height, width, 3), 18, np.uint8)
    top, bot, left = 9, height - 10, 22
    for val, lab in ((1.0, "1"), (0.5, ".5"), (0.0, "0")):
        yv = int(top + (1 - val) * (bot - top))
        cv2.line(g, (left, yv), (width - 3, yv), (45, 45, 55), 1)
        cv2.putText(g, lab, (2, yv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1, cv2.LINE_AA)
    if title:
        cv2.putText(g, title, (left + 6, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    for k in plot_idx:
        col = ctx.colors[ctx.classes[k]]
        poly = [(int(x), int(top + (1 - pr[k]) * (bot - top))) for x, pr in pts]
        if len(poly) >= 2:
            cv2.polylines(g, [np.array(poly, np.int32)], False, col, 1, cv2.LINE_AA)
        for (x, y) in poly:
            cv2.circle(g, (x, y), 2, col, -1, cv2.LINE_AA)
    return g


def _render_smoothing_viz(centers: dict, pass_frames: list[dict], files: dict, date_str: str,
                          out_path: Path, ctx: _Ctx, *, max_tiles: int = 41,
                          prob_window_sec: float | None = None, prob_window_kmax: int | None = None,
                          prob_window_tri: bool = False, errors: set | None = None,
                          truth: dict | None = None) -> bool:
    """Один конкат на ПРОХОД (визит). Все кадры прохода; изменённые (centers={имя:сглаж_класс}) —
    жёлтая рамка; зелёная — в окне жёлтого (w:номера); маджента-уголок — ошибка vs истины; красная — M>1;
    вертикальная линия — разрыв > радиуса окна. Подписи латиницей (cv2 не рисует кириллицу)."""
    import cv2
    import numpy as np
    center_names = set(centers)
    errors = errors or set()
    truth = truth or {}
    mark_names = center_names | errors
    idxs = [i for i, fr in enumerate(pass_frames) if fr["name"] in mark_names]
    if not idxs:
        return False
    seg = pass_frames
    if len(seg) > max_tiles:
        anchor = idxs[len(idxs) // 2]
        half = max_tiles // 2
        lo = max(0, min(anchor - half, len(seg) - max_tiles))
        seg = seg[lo: lo + max_tiles]
    cidx = [i for i, fr in enumerate(seg) if fr["name"] in center_names]

    win_owners: dict[int, list[int]] = {}
    if prob_window_sec and cidx:
        rk = (prob_window_kmax - 1) // 2 if prob_window_kmax else None
        rt = prob_window_sec / 2.0
        for c in cidx:
            lo = 0 if rk is None else max(0, c - rk)
            hi = len(seg) if rk is None else min(len(seg), c + rk + 1)
            tc = seg[c]["t"]
            for j in range(lo, hi):
                if abs(seg[j]["t"] - tc) <= rt:
                    win_owners.setdefault(j, []).append(c + 1)

    seg_sm = None
    if prob_window_sec:
        seg_P = np.array([fr["probs"] for fr in seg], dtype=float)
        seg_t = [fr["t"] for fr in seg]
        seg_sm = list(_ts.window_average(seg_P, seg_t, prob_window_sec,
                                         triangular=prob_window_tri, kmax=prob_window_kmax))

    TH, MAXW, LAB = 150, 150, 66
    sep_thr = prob_window_sec / 2.0 if prob_window_sec else None

    def _win_sep(h: int):
        bar = np.zeros((h, 7, 3), np.uint8)
        bar[:, 2:5] = 255
        return bar

    def _paint_seps(img, xs):
        for x in xs:
            if 0 <= x and x + 7 <= img.shape[1]:
                img[:, x:x + 7] = 0
                img[:, x + 2:x + 5] = 255

    plot_set: set[int] = set()   # классы для графика: те, что где-то argmax (модель или сглаженный)
    tiles, pts, xoff, sep_x = [], [], 0, []
    for k, fr in enumerate(seg):
        is_center = fr["name"] in center_names
        M = fr["M"]
        f = files.get(fr["name"])
        img = cv2.imread(str(f)) if f and Path(f).is_file() else None
        if img is None:
            img = np.full((TH, MAXW, 3), 40, np.uint8)
        else:
            h, w = img.shape[:2]
            nw = min(MAXW, max(1, int(w * (TH / h))))
            img = cv2.resize(img, (nw, TH))
        nw = img.shape[1]
        tw = MAXW
        canvas = np.full((TH + LAB, tw, 3), 25, np.uint8)
        x0 = (tw - nw) // 2
        canvas[:TH, x0:x0 + nw] = img
        hhmmss, _ymd = _fmt_dt(fr["t"], date_str)
        m_idx = int(np.argmax(fr["probs"]))
        if seg_sm is not None:
            s_idx = int(np.argmax(seg_sm[k]))
        elif fr["name"] in centers and centers[fr["name"]] in ctx.idx:
            s_idx = ctx.idx[centers[fr["name"]]]
        else:
            s_idx = m_idx
        plot_set.add(m_idx)
        plot_set.add(s_idx)
        base_txt = f"{k + 1} {hhmmss}"
        cv2.putText(canvas, base_txt, (2, TH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (210, 210, 210), 1, cv2.LINE_AA)
        owners = win_owners.get(k)
        if owners:
            (bw2, _), _ = cv2.getTextSize(base_txt + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(canvas, "(w:" + ",".join(map(str, owners)) + ")", (2 + bw2, TH + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 0), 1, cv2.LINE_AA)
        if M > 1:
            cv2.putText(canvas, f"M{M}", (tw - 30, TH + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (60, 60, 235), 1, cv2.LINE_AA)
        if s_idx != m_idx:
            sm_cls = centers.get(fr["name"], ctx.classes[s_idx])
            base = f"{ctx.short.get(ctx.classes[m_idx], '?')[:6]}->"
            cv2.putText(canvas, base, (2, TH + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (150, 150, 150), 1, cv2.LINE_AA)
            (bw, _), _ = cv2.getTextSize(base, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(canvas, ctx.short.get(sm_cls, sm_cls)[:6], (2 + bw, TH + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ctx.colors.get(sm_cls, (0, 215, 255)),
                        2 if is_center else 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, ctx.short.get(ctx.classes[s_idx], "?")[:6], (2, TH + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, ctx.colors[ctx.classes[s_idx]], 1, cv2.LINE_AA)
        _draw_probs(canvas, 2, TH + 50, fr["probs"], ctx, labels=True)
        if M > 1:
            cv2.rectangle(canvas, (0, 0), (tw - 1, TH - 1), (60, 60, 200), 4)
        if owners:
            cv2.rectangle(canvas, (6, 6), (tw - 7, TH - 7), (0, 210, 0), 3)
        if is_center:
            cv2.rectangle(canvas, (0, 0), (tw - 1, TH - 1), (0, 215, 255), 5)
        if fr["name"] in errors:
            tri = np.array([[0, 0], [30, 0], [0, 30]], np.int32)
            cv2.fillConvexPoly(canvas, tri, (200, 0, 200))
            tcls = next(iter(truth.get(fr["name"], [])), "?")
            cv2.putText(canvas, ctx.short.get(tcls, "?")[:5], (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(canvas)
        pts.append((xoff + tw / 2.0, fr["probs"]))
        xoff += tw
        if sep_thr is not None and k + 1 < len(seg) and (seg[k + 1]["t"] - fr["t"]) > sep_thr:
            sep = _win_sep(canvas.shape[0])
            sep_x.append(xoff)
            tiles.append(sep)
            xoff += sep.shape[1]

    strip = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
    plot_idx = sorted(plot_set)
    graph = _draw_prob_graph(strip.shape[1], pts, ctx, plot_idx, title="model probs" if prob_window_sec else "")
    graph2 = None
    if seg_sm is not None:
        sm_pts = [(pts[k][0], seg_sm[k]) for k in range(len(seg))]
        graph2 = _draw_prob_graph(strip.shape[1], sm_pts, ctx, plot_idx, title=f"smoothed (window {prob_window_sec:g}s)")

    HH = 40
    midx = [i for i, fr in enumerate(seg) if fr["name"] in mark_names]
    rep = seg[midx[0]]["name"]
    hdr = f"pass: {len(centers)} fixes, {len(errors)} err vs truth  (all frames shown, max {max_tiles})  {rep}"
    legend = "num time | yellow=fixed  magenta=err-vs-truth  red=M>1"
    if prob_window_sec:
        legend += "  green=in-window(w:owners)  |=win-break"
    (nw_, _), _ = cv2.getTextSize(hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    (lw_, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    W = max(strip.shape[1], nw_ + 12, lw_ + 12)
    header = np.full((HH, W, 3), 15, np.uint8)
    cv2.putText(header, hdr, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(header, legend, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    if W > strip.shape[1]:
        strip = cv2.hconcat([strip, np.full((strip.shape[0], W - strip.shape[1], 3), 25, np.uint8)])
        graph = cv2.hconcat([graph, np.full((graph.shape[0], W - graph.shape[1], 3), 18, np.uint8)])
        if graph2 is not None:
            graph2 = cv2.hconcat([graph2, np.full((graph2.shape[0], W - graph2.shape[1], 3), 18, np.uint8)])
    _paint_seps(graph, sep_x)
    if graph2 is not None:
        _paint_seps(graph2, sep_x)
    parts = [header, strip, graph] + ([graph2] if graph2 is not None else [])
    full = cv2.vconcat(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full)
    return True


# ─── ядро сглаживания одной даты ──────────────────────────────────────────────
def smooth_date(date_dir: Path, cfg: SmoothCfg, *, gap: float, p_stay: float, gate: float | None,
                include_uncertain: bool = True, max_persons: int = 1, ext: str = "jpg",
                viz: bool = False, viz_dir: Path | None = None, out_dir: Path | None = None,
                guard_sec: float | None = None, min_minority_prob: float | None = None,
                veto_prob: float | None = None, prob_window_sec: float | None = None,
                prob_window_kmax: int | None = None, prob_window_tri: bool = False,
                merge_zones: bool = False, hard_vote: bool = False,
                truth: dict | None = None, version: str = "") -> dict:
    """Сглаживает одну дату. НЕ мутирует images/ — пишет сайдкар в out_dir (default smoothed/<date>/)."""
    rows = _read_csv_rows(date_dir, cfg.csv_name)
    if not rows:
        return {"date": date_dir.name, "rows": 0}
    ctx = _resolve_ctx(cfg, list(rows[0].keys()))
    if not ctx.classes:
        return {"date": date_dir.name, "rows": len(rows), "eligible": 0}

    if out_dir is None:
        out_dir = date_dir.parents[1] / "smoothed" / date_dir.name

    cfg_sig = (f"{version}|{cfg.smoother}|c{len(ctx.classes)}|gap{gap}|ps{p_stay}|gate{gate}"
               f"|pw{prob_window_sec}|k{prob_window_kmax}|tri{int(prob_window_tri)}|mz{int(merge_zones)}"
               f"|hv{int(hard_vote)}|t{len(truth or {})}")
    wm_path = out_dir / "smooth_state.json"
    if wm_path.is_file() and (out_dir / "classifications_smoothed.csv").is_file():
        try:
            _wm = json.loads(wm_path.read_text(encoding="utf-8"))
            if _wm.get("csv_rows") == len(rows) and _wm.get("sig") == cfg_sig:
                return {"date": date_dir.name, "rows": len(rows), "eligible": _wm.get("eligible", 0),
                        "changed": 0, "moved": 0, "rescued_uncertain": 0, "to_next": 0,
                        "viz": 0, "errors": 0, "errors_unfixed": 0, "skipped": True}
        except (ValueError, OSError):
            pass

    files = _index_files(date_dir, ext)
    recs: list[dict] = []
    meta: dict[str, dict] = {}
    for r in rows:
        name = r.get("crop", "")
        if not name:
            continue
        pr = _parse_cam_t(name, merge_zones)
        if pr is None:
            continue
        cam, t = pr
        try:
            te = r.get("ts_epoch", "")
            if te:
                t = float(te)
        except ValueError:
            pass
        try:
            probs = [float(r.get(f"p_{c}", 0) or 0) for c in ctx.classes]
        except ValueError:
            continue
        if name in meta:
            continue
        rec = {"cam": cam, "t": t, "probs": probs, "name": name}
        meta[name] = rec
        recs.append(rec)

    if not recs:
        _write_watermark(out_dir, csv_rows=len(rows), eligible=0, changed=0, sig=cfg_sig, smoother=cfg.smoother)
        return {"date": date_dir.name, "rows": len(rows), "eligible": 0}

    smoothed = _ts.smooth_sequence(
        recs, ctx.classes, gap_sec=gap, p_stay=p_stay, gate=gate,
        guard_sec=guard_sec, protect_class=ctx.resident_class,
        minority_classes=[c for c in ctx.classes if c != ctx.resident_class],
        min_minority_prob=min_minority_prob, context=[], veto_prob=veto_prob,
        prob_window_sec=prob_window_sec, prob_window_kmax=prob_window_kmax, prob_window_tri=prob_window_tri,
        hard_vote=hard_vote)

    viz_n = 0
    audit, corrections, viz_targets = [], [], []
    smoothed_by_name: dict[str, str] = {}
    for i, (rec, (sm_class, was_changed)) in enumerate(zip(recs, smoothed)):
        name = rec["name"]
        f = files.get(name)
        if f is None or not f.is_file():
            continue
        smoothed_by_name[name] = sm_class
        rel = f.relative_to(date_dir).as_posix()
        cur = _current_class(f.relative_to(date_dir).parts)
        # rel + p_* добавляем ПОСЛЕ changed: col4 остаётся smoothed_class (identify.sh читает $4)
        row_out = {"crop": name, "cam": rec["cam"], "model_class": cur,
                   "smoothed_class": sm_class, "changed": cur != sm_class, "rel": rel}
        for c, pv in zip(ctx.classes, rec["probs"]):
            row_out[f"p_{c}"] = round(pv, 4)
        audit.append(row_out)
        if was_changed:
            viz_targets.append((i, name, sm_class))
        if cur != sm_class:
            corrections.append((name, cur, sm_class))

    changed_set = {name for (_, name, _) in viz_targets}
    errors = {n: sm for n, sm in smoothed_by_name.items()
              if truth and truth.get(n) and sm not in truth[n]}
    n_err_unfixed = sum(1 for n in errors if n not in changed_set)

    if viz and (viz_targets or errors):
        vdir = viz_dir or (out_dir / "smooth_viz")
        sig = (f"{len(rows)}:{version}:pw{prob_window_sec}:k{prob_window_kmax}:tri{int(prob_window_tri)}"
               f":hv{int(hard_vote)}:mz{int(merge_zones)}:zn2:t{len(errors)}")   # mz/zn2 → перерисовать конкаты при смене склейки/именования
        marker = vdir / ".viz_rows"
        prev = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if prev == sig and any(vdir.glob("*.jpg")):
            pass
        else:
            vdir.mkdir(parents=True, exist_ok=True)
            for _old in vdir.glob("*.jpg"):
                _old.unlink()
            cam_frames = _frames_by_cam(rows, ctx, merge_zones)
            changed_names = {name: sm for (_, name, sm) in viz_targets}
            interesting = set(changed_names) | set(errors)
            for cam, frs in cam_frames.items():
                times = [f["t"] for f in frs]
                for vis in _ts.segment_visits(times, gap):
                    vis_names = [frs[j]["name"] for j in vis]
                    if not any(n in interesting for n in vis_names):
                        continue
                    centers = {n: changed_names[n] for n in vis_names if n in changed_names}
                    errs = {n for n in vis_names if n in errors}
                    pass_frames = [frs[j] for j in vis]
                    rep = min(n for n in vis_names if n in interesting)
                    rep_stem = Path(rep).stem
                    if merge_zones:   # merged-конкат: общий суффикс зон (напр. cam_01_9_du), если в проходе обе
                        _zs = sorted({z for n in vis_names for z in [_split_zone(Path(n).stem)[1]] if z})
                        if len(_zs) > 1:
                            _pre, _z, _suf = _split_zone(rep_stem)
                            if _pre is not None:
                                rep_stem = "_".join(_pre + ["".join(_zs)] + _suf)
                    try:
                        if _render_smoothing_viz(centers, pass_frames, files, date_dir.name,
                                                 vdir / f"{rep_stem}.jpg", ctx,
                                                 prob_window_sec=prob_window_sec, prob_window_kmax=prob_window_kmax,
                                                 prob_window_tri=prob_window_tri, errors=errs, truth=truth):
                            viz_n += 1
                    except Exception as e:
                        logger.warning("viz fail %s: %s", rep, e)
            marker.write_text(sig, encoding="utf-8")

    changed = len(corrections)
    rescued = sum(1 for (_n, cur, _sm) in corrections if cur in ("uncertain", "unknown_resident"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (["crop", "cam", "model_class", "smoothed_class", "changed", "rel"]
                  + [f"p_{c}" for c in ctx.classes])
    with open(out_dir / "classifications_smoothed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(audit)
    # labels.json сглаживатель НЕ пишет: сглаженные классы уже в classifications_smoothed.csv
    # (col4 smoothed_class). labels.json — файл РУЧНОЙ разметки: его пишет ТОЛЬКО человек в 2_label_ui
    # (разметчик сидит стартовые метки из CSV, если labels.json нет), поэтому пересчёт CSV сколько
    # угодно раз не затирает ручную разметку.
    _write_watermark(out_dir, csv_rows=len(rows), eligible=len(recs), changed=changed, sig=cfg_sig, smoother=cfg.smoother)
    n_next = sum(1 for a in audit if a["smoothed_class"] in ctx.downstream_classes)
    return {"date": date_dir.name, "rows": len(rows), "eligible": len(recs), "changed": changed,
            "moved": 0, "rescued_uncertain": rescued, "to_next": n_next, "viz": viz_n,
            "errors": len(errors), "errors_unfixed": n_err_unfixed}


def _date_dirs(root: Path) -> list[Path]:
    if _DATE_RE.match(root.name):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _DATE_RE.match(d.name))


def _load_truth(path: Path, classes: set | None = None) -> dict:
    if not path.exists():
        logger.warning("[truth] не найдено: %s", path)
        return {}
    files = [path] if path.is_file() else sorted(path.rglob("labels.json"))
    truth: dict[str, set] = {}
    for lf in files:
        try:
            labs = _ml.load_labels(lf)
        except Exception as e:
            logger.warning("[truth] не прочитан %s: %s", lf, e)
            continue
        for k, v in labs.items():
            cls = {c for c in (v or []) if classes is None or c in classes}
            if cls:
                truth[Path(k).name] = cls
    logger.info("[truth] загружено %d меток из %d файлов", len(truth), len(files))
    return truth


def run_service(cfg: SmoothCfg, *, version: str = "", desc: str = "") -> int:
    """CLI + watch-цикл сглаживания для заданного конфига (используют драйверы 3b/4b)."""
    setup_logging()
    ap = argparse.ArgumentParser(description=desc or f"Темпоральное сглаживание ({cfg.smoother})")
    ap.add_argument("input_dir", type=Path, help="images/ (все даты) или конкретный каталог-дата")
    ap.add_argument("--gap", type=float, default=_ts.DEFAULT_GAP_SEC)
    ap.add_argument("--p-stay", type=float, default=_ts.DEFAULT_P_STAY)
    ap.add_argument("--gate", type=float, default=0.8, help="не трогать уверенные ≥ gate (0=выкл)")
    ap.add_argument("--max-persons", type=int, default=1)
    ap.add_argument("--no-uncertain", dest="include_uncertain", action="store_false")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--prob-window", type=float, default=0.0, help="бегущее окно ±N/2 сек вместо Viterbi (0=Viterbi)")
    ap.add_argument("--prob-window-k", type=int, default=0)
    ap.add_argument("--prob-window-tri", action="store_true")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=120.0)
    # флаги-режимы: дефолт False, управление через .env → _build_args в 3b/4b .sh (передают флаг per-poll).
    # НЕ брать env как argparse-default: store_true не выключается из CLI, а .sh экспортирует env лишь при
    # старте → stale-значение мешало бы выключить флаг без рестарта.
    ap.add_argument("--merge-zones", action="store_true",
                    help="склеивать зоны d/u ОДНОЙ камеры в один поток по времени (по умолч. ВЫКЛ; "
                         "в проде через .env SMOOTH_MERGE_ZONES). С Viterbi эффект ~в пределах шума")
    ap.add_argument("--hard-vote", action="store_true",
                    help="ЖЁСТКИЙ ГОЛОС: 1 класс на весь визит = argmax среднего probs (вместо Viterbi/окна; "
                         "в проде через .env SMOOTH_HARD_VOTE, по умолч. ВЫКЛ). ⚠️ opt-in для экспериментов: "
                         "на независимой истине хуже Viterbi и топит меньшинства (см. docs/ml.md)")
    args = ap.parse_args()

    if not args.input_dir.exists():
        logger.error("Не найдено: %s", args.input_dir)
        return 1
    gate = args.gate if args.gate and args.gate > 0 else None
    prob_window = args.prob_window if args.prob_window and args.prob_window > 0 else None
    prob_window_k = args.prob_window_k if args.prob_window_k and args.prob_window_k > 0 else None
    truth = _load_truth(args.truth, set(cfg.fixed_classes) if cfg.fixed_classes else None) if args.truth else None
    logger.info("%s smooth: gap=%.0fс p_stay=%.2f gate=%s viz=%s prob_window=%s k=%s tri=%s merge_zones=%s hard_vote=%s truth=%s",
                cfg.smoother, args.gap, args.p_stay, gate, args.viz, prob_window, prob_window_k,
                args.prob_window_tri, args.merge_zones, args.hard_vote, len(truth) if truth else 0)

    while True:
        total_changed = n_skip = 0
        for dd in _date_dirs(args.input_dir):
            st = smooth_date(dd, cfg, gap=args.gap, p_stay=args.p_stay, gate=gate,
                             include_uncertain=args.include_uncertain, max_persons=args.max_persons,
                             ext=args.ext, viz=args.viz, prob_window_sec=prob_window,
                             prob_window_kmax=prob_window_k, prob_window_tri=args.prob_window_tri,
                             merge_zones=args.merge_zones, hard_vote=args.hard_vote,
                             truth=truth, version=version)
            if st.get("skipped"):
                n_skip += 1
            if st.get("changed") or st.get("errors"):
                logger.info("1 батч (%d кадров)  Готово.  %s: сглажено %d (rescued %d, to_next %d, "
                            "viz %d, ошибок vs истины %d/неиспр %d)",
                            st.get("eligible", 0), st["date"], st["changed"], st["rescued_uncertain"],
                            st["to_next"], st.get("viz", 0), st.get("errors", 0), st.get("errors_unfixed", 0))
                total_changed += st["changed"]
        if total_changed == 0:
            logger.info("(%s) Изменений нет (%d дат пропущено) — ожидание %.0fs…", cfg.smoother, n_skip, args.poll_sec)
        if args.once:
            break
        time.sleep(args.poll_sec)
    return 0
