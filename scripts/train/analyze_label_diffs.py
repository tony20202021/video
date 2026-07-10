"""Анализ pixel-diff внутри одного человека vs между разными людьми.

Читает labels.json из каталога датасета, вычисляет mean-abs-diff между
соседними кадрами в каждой (cam, date)-группе, разделяет на:
  - intra: оба кадра принадлежат одному person_id
  - inter: кадры принадлежат разным person_id

Показывает:
  1. Гистограммы intra vs inter
  2. Оценку feasibility полу-автоматической разметки по порогу дифа

Usage:
    python scripts/train/analyze_label_diffs.py .data/residents/v0/new
    python scripts/train/analyze_label_diffs.py .data/residents/v0/new --out .data/residents/v0/analysis
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_RE_FRAME = re.compile(
    r"^(?P<cam>cam_[^_]+)_\d+_[a-z]_(?P<date>\d{8})_(?P<time>\d{6})_\d+",
    re.IGNORECASE,
)


def _parse(name: str) -> tuple[str, str, int] | None:
    m = _RE_FRAME.match(name)
    if not m:
        return None
    t = m.group("time")
    sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return m.group("cam"), m.group("date"), sod


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]
    if h1 != h2 or w1 != w2:
        b = cv2.resize(b, (w1, h1))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def analyze(data_dir: Path, out_dir: Path) -> dict:
    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json не найден: {labels_path}")

    raw = json.loads(labels_path.read_text())
    labels: dict[str, str] = {
        Path(k).name: v
        for k, v in raw.get("labels", raw).items()
        if v != "skip"
    }

    # Парсим и группируем по (cam, date)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for name, pid in labels.items():
        parsed = _parse(name)
        if parsed is None:
            continue
        cam, date, sod = parsed
        groups[(cam, date)].append({"name": name, "pid": pid, "sod": sod})

    for key in groups:
        groups[key].sort(key=lambda x: x["sod"])

    # Вычисляем дифы между соседними кадрами
    intra: list[float] = []   # same person
    inter: list[float] = []   # different persons
    inter_pairs: list[tuple[str, str, float]] = []  # (pid_a, pid_b, diff)

    for (cam, date), items in sorted(groups.items()):
        prev_bgr: np.ndarray | None = None
        prev_pid: str | None = None

        for item in items:
            path = data_dir / item["name"]
            if not path.exists():
                prev_bgr = None
                prev_pid = None
                continue

            bgr = cv2.imread(str(path))
            if bgr is None:
                prev_bgr = None
                prev_pid = None
                continue

            if prev_bgr is not None and prev_pid is not None:
                d = _diff(prev_bgr, bgr)
                if item["pid"] == prev_pid:
                    intra.append(d)
                else:
                    inter.append(d)
                    inter_pairs.append((prev_pid, item["pid"], d))

            prev_bgr = bgr
            prev_pid = item["pid"]

    intra_arr = np.array(intra) if intra else np.array([])
    inter_arr = np.array(inter) if inter else np.array([])

    print(f"\nIntra-person дифы (одинаковый человек): n={len(intra_arr)}")
    if len(intra_arr):
        print(f"  min={intra_arr.min():.1f}  median={np.median(intra_arr):.1f}"
              f"  p75={np.percentile(intra_arr,75):.1f}"
              f"  p90={np.percentile(intra_arr,90):.1f}"
              f"  max={intra_arr.max():.1f}")

    print(f"\nInter-person дифы (разные люди): n={len(inter_arr)}")
    if len(inter_arr):
        print(f"  min={inter_arr.min():.1f}  median={np.median(inter_arr):.1f}"
              f"  p10={np.percentile(inter_arr,10):.1f}"
              f"  p25={np.percentile(inter_arr,25):.1f}"
              f"  max={inter_arr.max():.1f}")

    # Поиск оптимального порога для полу-авто разметки
    print("\n=== Feasibility полу-автоматической разметки ===")
    if len(intra_arr) and len(inter_arr):
        thresholds = np.arange(5, 120, 2.5)
        best_threshold = None
        best_score = -1.0

        rows = []
        for t in thresholds:
            tp = float(np.sum(intra_arr <= t))    # intra правильно ≤ t → same person
            fn = float(np.sum(intra_arr > t))     # intra пропущено
            tn = float(np.sum(inter_arr > t))     # inter правильно > t → boundary
            fp = float(np.sum(inter_arr <= t))    # inter неправильно ≤ t (смешали людей)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Для разметки: важнее не смешать людей (precision), чем покрыть всё (recall)
            # Штраф за fp: помечаем разного человека как того же → ошибка разметки
            contamination = fp / (fp + tn) if (fp + tn) > 0 else 0

            rows.append((t, tp, fn, fp, tn, precision, recall, f1, contamination))
            if f1 > best_score:
                best_score = f1
                best_threshold = t

        print(f"\n{'Порог':>6}  {'TP':>5}  {'FN':>5}  {'FP':>5}  {'TN':>5}"
              f"  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'Contam':>6}")
        for t, tp, fn, fp, tn, pr, re_, f1, cont in rows:
            marker = " ←" if t == best_threshold else ""
            print(f"  {t:5.1f}  {tp:5.0f}  {fn:5.0f}  {fp:5.0f}  {tn:5.0f}"
                  f"  {pr:5.3f}  {re_:5.3f}  {f1:5.3f}  {cont:6.3f}{marker}")

        print(f"\nОптимальный порог (max F1): {best_threshold:.1f}")

        # При разных порогах с разными приоритетами
        print("\nПорог при contamination ≤ 1%  (почти не смешиваем людей):")
        for t, tp, fn, fp, tn, pr, re_, f1, cont in rows:
            if cont <= 0.01:
                pct_auto = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                print(f"  threshold={t:.1f}: авторазметка {pct_auto:.0f}% intra-кадров, "
                      f"precision={pr:.3f}, contamination={cont:.3f}")

        print("\nПорог при contamination ≤ 5%:")
        for t, tp, fn, fp, tn, pr, re_, f1, cont in rows:
            if cont <= 0.05:
                pct_auto = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                print(f"  threshold={t:.1f}: авторазметка {pct_auto:.0f}% intra-кадров, "
                      f"precision={pr:.3f}, contamination={cont:.3f}")
                break

    # Строим график
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Гистограммы intra vs inter
        ax = axes[0]
        all_vals = np.concatenate([intra_arr, inter_arr]) if len(intra_arr) and len(inter_arr) else np.array([])
        bins = np.linspace(0, min(all_vals.max(), 200) if len(all_vals) else 200, 50)
        if len(intra_arr):
            ax.hist(intra_arr, bins=bins, alpha=0.6, color="steelblue",
                    label=f"intra (same person, n={len(intra_arr)})", density=True)
        if len(inter_arr):
            ax.hist(inter_arr, bins=bins, alpha=0.6, color="tomato",
                    label=f"inter (diff person, n={len(inter_arr)})", density=True)
        ax.set_xlabel("mean-abs pixel diff")
        ax.set_ylabel("density")
        ax.set_title("Intra vs Inter person pixel diff")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # F1 / contamination vs threshold
        ax2 = axes[1]
        if len(intra_arr) and len(inter_arr):
            ts = [r[0] for r in rows]
            f1s = [r[7] for r in rows]
            conts = [r[8] for r in rows]
            ax2.plot(ts, f1s, "b-", label="F1 (intra/inter boundary)")
            ax2.plot(ts, conts, "r--", label="Contamination (inter смешано)")
            ax2.axhline(0.05, color="gray", linestyle=":", alpha=0.7, label="5% contamination")
            ax2.axhline(0.01, color="lightgray", linestyle=":", alpha=0.7, label="1% contamination")
            if best_threshold is not None:
                ax2.axvline(best_threshold, color="blue", linestyle="--", alpha=0.5,
                            label=f"best threshold={best_threshold:.0f}")
            ax2.set_xlabel("diff threshold")
            ax2.set_ylabel("score")
            ax2.set_title("Feasibility: F1 и contamination vs порог")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)

        fig.tight_layout()
        out_path = out_dir / "label_diff_analysis.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\nГрафик: {out_path}")
    except Exception as e:
        print(f"  [!] Matplotlib: {e}")

    return {
        "n_intra": len(intra_arr),
        "n_inter": len(inter_arr),
        "intra_median": float(np.median(intra_arr)) if len(intra_arr) else None,
        "inter_median": float(np.median(inter_arr)) if len(inter_arr) else None,
        "best_threshold": best_threshold if (len(intra_arr) and len(inter_arr)) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v0" / "analysis")
    args = ap.parse_args()
    analyze(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
