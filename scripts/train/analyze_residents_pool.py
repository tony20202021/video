"""Анализ полного пула кадров для датасета жителей (до дедупликации).

Строит три группы графиков:
  1. frames_per_window   — сколько кадров в одном N-секундном окне
  2. conf_distribution   — распределение confidence по выборке и по окнам
  3. pixel_diff          — разность между соседними кадрами в одном окне

Usage:
    python scripts/train/analyze_residents_pool.py
    python scripts/train/analyze_residents_pool.py --interval 1
    python scripts/train/analyze_residents_pool.py --out .data/residents/v0/analysis
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASSES = ["1_resident", "4_guest"]

_RE_FRAME = re.compile(
    r"^(?P<cam>cam_[^_]+)_\d+_[a-z]_(?P<date>\d{8})_(?P<time>\d{6})_\d+_.*_conf(?P<conf>\d+\.\d+)",
    re.IGNORECASE,
)


def _parse(name: str) -> tuple[str, str, int, float] | None:
    m = _RE_FRAME.match(name)
    if not m:
        return None
    t = m.group("time")
    sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return m.group("cam"), m.group("date"), sod, float(m.group("conf"))


def _default_sources(classes: list[str] = DEFAULT_CLASSES) -> list[Path]:
    base = REPO_ROOT / ".data" / "groups" / "v1"
    sources: list[Path] = []
    for cls in classes:
        d = base / "dataset" / cls
        if d.is_dir():
            sources.append(d)
    inference_root = base / "inference" / "images"
    if inference_root.is_dir():
        for date_dir in sorted(inference_root.iterdir()):
            for cls in classes:
                d = date_dir / cls
                if d.is_dir():
                    sources.append(d)
    return sources


def load_pool(sources: list[Path]) -> list[tuple[Path, str, str, int, float]]:
    pool = []
    seen: set[str] = set()
    for src_dir in sources:
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            parsed = _parse(f.stem)
            if parsed:
                cam, date, sod, conf = parsed
                pool.append((f, cam, date, sod, conf))
    return pool


def _make_windows(pool: list, interval: int) -> dict[tuple, list[tuple[int, float, Path]]]:
    """Groups pool entries into fixed-width windows. Each entry: (sod, conf, path)."""
    windows: dict[tuple, list] = defaultdict(list)
    for path, cam, date, sod, conf in pool:
        key = (cam, date, sod // interval)
        windows[key].append((sod, conf, path))
    return windows


# ── Plot 1: frames per window ──────────────────────────────────────────────────

def plot_frames_per_window(pool: list, interval: int, out_dir: Path
                           ) -> dict[tuple, list]:
    windows = _make_windows(pool, interval)
    counts = np.array([len(v) for v in windows.values()])
    n_windows = len(windows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Кадров в {interval}-секундном окне  "
        f"(окон: {n_windows}, кадров всего: {len(pool)})",
        fontsize=13,
    )

    ax = axes[0]
    max_c = int(counts.max())
    bins = np.arange(0.5, max_c + 1.5, 1)
    ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(counts.mean(), color="orange", linestyle="--",
               label=f"Среднее: {counts.mean():.2f}")
    ax.set_xlabel("Кадров в окне")
    ax.set_ylabel("Количество окон")
    ax.set_title("Распределение числа кадров")
    ax.legend()

    ax2 = axes[1]
    ax2.axis("off")
    single = int((counts == 1).sum())
    multi  = int((counts > 1).sum())
    kept   = n_windows
    disc   = len(pool) - kept
    text = (
        f"Минимум:            {int(counts.min())}\n"
        f"Максимум:           {int(counts.max())}\n"
        f"Среднее:            {counts.mean():.2f}\n"
        f"Медиана:            {float(np.median(counts)):.1f}\n\n"
        f"Окон с 1 кадром:    {single} ({100*single/n_windows:.0f}%)\n"
        f"Окон с 2+ кадрами:  {multi} ({100*multi/n_windows:.0f}%)\n\n"
        f"Отбирается (1/окно): {kept}\n"
        f"Отброшено:           {disc} ({100*disc/len(pool):.0f}%)"
    )
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

    plt.tight_layout()
    out = out_dir / f"1_frames_per_{interval}s_window.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  {out}")
    return windows


# ── Plot 2: confidence distribution ───────────────────────────────────────────

def plot_conf_distribution(pool: list, windows: dict, interval: int, out_dir: Path):
    all_confs = np.array([conf for _, _, _, _, conf in pool])
    window_max = np.array([max(c for _, c, _ in v) for v in windows.values()])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Распределение confidence", fontsize=13)

    # 1: Overall histogram
    ax = axes[0, 0]
    ax.hist(all_confs, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    for thr, col in zip([0.6, 0.7, 0.8, 0.9], ["gold", "orange", "tomato", "crimson"]):
        n = int((all_confs >= thr).sum())
        ax.axvline(thr, linestyle="--", color=col, label=f"≥{thr}: {n} кадров")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Кадров")
    ax.set_title("Все кадры (до дедупликации)")
    ax.legend(fontsize=9)

    # 2: Windows surviving threshold on max conf
    ax = axes[0, 1]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    kept_windows = [int((window_max >= t).sum()) for t in thresholds]
    bars = ax.bar([str(t) for t in thresholds], kept_windows, color="steelblue", width=0.6)
    ax.set_xlabel("Порог confidence")
    ax.set_ylabel("Окон (≥ порога)")
    ax.set_title("Окна где max(conf) ≥ порога\n(отбираем 1 лучший из каждого окна)")
    for bar, v in zip(bars, kept_windows):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=9)

    # 3: Frames within delta of window max
    ax = axes[1, 0]
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    frames_by_delta = []
    for delta in deltas:
        total = sum(
            sum(1 for _, c, _ in items if c >= max(c2 for _, c2, _ in items) - delta)
            for items in windows.values()
        )
        frames_by_delta.append(total)
    bars = ax.bar([str(d) for d in deltas], frames_by_delta, color="coral", width=0.6)
    ax.axhline(len(pool), linestyle="--", color="gray", label=f"Всего кадров: {len(pool)}")
    ax.set_xlabel("Δ от max(conf) в окне")
    ax.set_ylabel("Кадров")
    ax.set_title("Кадров если брать conf ≥ max(окна) − Δ\n(вместо только одного максимального)")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, frames_by_delta):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=9)

    # 4: Histogram of window max confs
    ax = axes[1, 1]
    ax.hist(window_max, bins=40, color="mediumseagreen", edgecolor="white", linewidth=0.5)
    ax.axvline(window_max.mean(), color="orange", linestyle="--",
               label=f"Среднее: {window_max.mean():.2f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Окон")
    ax.set_title(f"Распределение max(conf) по {interval}-секундным окнам")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = out_dir / f"2_conf_distribution_{interval}s.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  {out}")


# ── Plot 3: pixel diff ─────────────────────────────────────────────────────────

def plot_pixel_diff(windows: dict, out_dir: Path, interval: int, max_pairs: int = 3000):
    multi = {k: v for k, v in windows.items() if len(v) > 1}
    print(f"  Окон с 2+ кадрами: {len(multi)}, вычисляем попарные разности...")

    diffs: list[float] = []
    pairs_done = 0

    for items in sorted(multi.values(), key=lambda v: v[0][0]):
        sorted_items = sorted(items, key=lambda x: x[0])  # by sod
        paths = [p for _, _, p in sorted_items]
        for i in range(len(paths) - 1):
            if pairs_done >= max_pairs:
                break
            img1 = cv2.imread(str(paths[i]))
            img2 = cv2.imread(str(paths[i + 1]))
            if img1 is None or img2 is None:
                continue
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            diff = float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))
            diffs.append(diff)
            pairs_done += 1
        if pairs_done >= max_pairs:
            break

    if not diffs:
        print("  Нет пар для сравнения")
        return

    diffs_arr = np.array(diffs)
    print(f"  Вычислено пар: {len(diffs_arr)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Попиксельная разность между соседними кадрами в одном окне  (n={len(diffs_arr)} пар)",
        fontsize=13,
    )

    ax = axes[0]
    ax.hist(diffs_arr, bins=60, color="mediumpurple", edgecolor="white", linewidth=0.5)
    for thr, col in zip([5, 10, 15, 20, 30], ["gold", "orange", "tomato", "crimson", "darkred"]):
        n = int((diffs_arr >= thr).sum())
        ax.axvline(thr, linestyle="--", color=col,
                   label=f"≥{thr}: {n} пар ({100*n/len(diffs_arr):.0f}%)")
    ax.set_xlabel("Средняя абс. разность (0–255)")
    ax.set_ylabel("Пар кадров")
    ax.set_title("Распределение pixel diff")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.axis("off")
    text = (
        f"Минимум:   {diffs_arr.min():.1f}\n"
        f"Максимум:  {diffs_arr.max():.1f}\n"
        f"Среднее:   {diffs_arr.mean():.1f}\n"
        f"Медиана:   {float(np.median(diffs_arr)):.1f}\n"
        f"P25:       {float(np.percentile(diffs_arr, 25)):.1f}\n"
        f"P75:       {float(np.percentile(diffs_arr, 75)):.1f}\n\n"
        "Идея: если брать из каждого окна\n"
        "только кадры отличающиеся\n"
        "от предыдущего > порога,\n"
        "можно взять несколько разных\n"
        "кадров из одного окна."
    )
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.6))

    plt.tight_layout()
    out = out_dir / f"3_pixel_diff_{interval}s.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Анализ пула кадров для датасета жителей")
    ap.add_argument("--interval", type=int, default=2, metavar="SEC",
                    help="Размер временного окна (default: 2)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, metavar="CLS")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v0" / "analysis",
                    help="Каталог для сохранения графиков")
    ap.add_argument("--max-pairs", type=int, default=3000,
                    help="Макс. пар для pixel diff (default: 3000)")
    args = ap.parse_args()

    sources = _default_sources(args.classes)
    print(f"Источники ({len(sources)}):")
    for s in sources:
        n = (sum(1 for f in s.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
             if s.is_dir() else 0)
        print(f"  {s}  [{n} файлов]")

    print("\nЗагружаем пул...")
    pool = load_pool(sources)
    if not pool:
        print("[!] Нет файлов с распознанными именами", file=sys.stderr)
        return 1
    print(f"Кадров с парсингом: {len(pool)}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Каталог графиков: {args.out}\n")

    print(f"[1] Кадров в {args.interval}-секундных окнах...")
    windows = plot_frames_per_window(pool, args.interval, args.out)

    print(f"\n[2] Распределение confidence...")
    plot_conf_distribution(pool, windows, args.interval, args.out)

    print(f"\n[3] Pixel diff между соседними кадрами...")
    plot_pixel_diff(windows, args.out, args.interval, args.max_pairs)

    print(f"\nГотово. Графики в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
