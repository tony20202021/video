"""
Анализ распределения значений diff при срабатывании порога движения.

Читает каталог с результатами 5_diff_yolo_boxes_low.py, извлекает diff из имён файлов
(_low_diff{N}.jpg, _raw_diff{N}.jpg), строит гистограмму и сохраняет отчёт.

Вход:  каталог run_* из .output/cameras/5_diff_yolo_boxes_low/
Выход: .output/bench/diff_analysis/run_*/
  diff_report.json   — статистика по категориям
  hist_all.txt       — ASCII гистограмма (всегда)
  hist_all.png       — PNG гистограмма (если matplotlib доступен)

Использование:
  python scripts/bench/4_diff_analysis.py
  python scripts/bench/4_diff_analysis.py --input .output/cameras/5_diff_yolo_boxes_low/run_20260601_094831_msk
  python scripts/bench/4_diff_analysis.py --input .output/cameras/5_diff_yolo_boxes_low/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMERAS_DIR = REPO_ROOT / ".output" / "cameras" / "5_diff_yolo_boxes_low"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "bench" / "4_diff_analysis"


def _latest_run(base: Path) -> Path | None:
    """Возвращает последний run_* каталог (по имени), или None."""
    runs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    return runs[-1] if runs else None

sys.path.insert(0, str(REPO_ROOT / "src"))
from common.utils.time_msk import ts_for_dir

_RE_DIFF = re.compile(r"_diff([\d.]+)\.jpg$", re.IGNORECASE)


_RE_TS = re.compile(r"_(\d{8}_\d{6})_\d+_msk")


def _ts_sec(fname: str) -> int | None:
    """Извлекает секунды (int) из имени файла: YYYYMMDD_HHMMSS → int."""
    m = _RE_TS.search(fname)
    if not m:
        return None
    s = m.group(1)  # "20260601_095151"
    try:
        from datetime import datetime as _dt
        return int(_dt.strptime(s, "%Y%m%d_%H%M%S").timestamp())
    except Exception:
        return None


def collect_diffs(input_dir: Path) -> dict[str, list[float]]:
    """Собирает все diff-значения из имён файлов в категории.

    Категории:
      low_diff   — срабатывание LOW, HI недоступен (фоновый шум + люди)
      raw_diff   — срабатывание HI (фоновый шум + люди)
      detected   — diff в момент обнаружения ЧЕЛОВЕКА (берём из соседнего raw-файла)
    """
    cats: dict[str, list[float]] = {
        "low_diff": [],
        "raw_diff": [],
        "detected": [],
    }

    runs = (
        [input_dir]
        if (input_dir / "raw").exists() or any(input_dir.glob("*.jpg"))
        else sorted(input_dir.iterdir())
    )

    for run_dir in runs:
        if not run_dir.is_dir():
            continue

        # Строим индекс: секунда → список (diff, имя) для raw-файлов
        raw_dir = run_dir / "raw"
        raw_index: dict[int, list[tuple[float, str]]] = {}  # ts_sec → [(diff, name)]
        if raw_dir.is_dir():
            for f in raw_dir.glob("*.jpg"):
                m = _RE_DIFF.search(f.name)
                if not m:
                    continue
                val = float(m.group(1))
                ts = _ts_sec(f.name)
                if "_low_diff" in f.name:
                    cats["low_diff"].append(val)
                elif "_raw_diff" in f.name:
                    cats["raw_diff"].append(val)
                if ts is not None:
                    raw_index.setdefault(ts, []).append((val, f.name))

        # Детекции людей → ищем ближайший raw-diff (±3 сек)
        for f in run_dir.glob("*.jpg"):
            if not (re.search(r"_p\d+\.jpg$", f.name) or re.search(r"_low_p\d+\.jpg$", f.name)):
                continue
            det_ts = _ts_sec(f.name)
            if det_ts is None:
                continue
            best_diff: float | None = None
            best_dist = 4
            for delta in range(-3, 4):
                for diff_val, _ in raw_index.get(det_ts + delta, []):
                    if abs(delta) < best_dist:
                        best_dist = abs(delta)
                        best_diff = diff_val
            if best_diff is not None:
                cats["detected"].append(best_diff)

    return cats


def ascii_histogram(values: list[float], bins: int = 15, width: int = 50) -> str:
    if not values:
        return "(нет данных)"
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    lines.append(f"  min={arr.min():.1f}  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
                 f"p75={np.percentile(arr,75):.1f}  p95={np.percentile(arr,95):.1f}  max={arr.max():.1f}")
    lines.append(f"  {'range':>12}  {'count':>6}  bar")
    for i, cnt in enumerate(counts):
        bar = "█" * int(cnt / max_count * width)
        lines.append(f"  [{edges[i]:6.1f},{edges[i+1]:6.1f})  {cnt:6d}  {bar}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Анализ распределения diff при срабатывании порога")
    parser.add_argument("--input",  type=Path, default=None,
                        help="Каталог run_* или родительский каталог с run_*/")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bins",   type=int, default=20)
    args = parser.parse_args()

    if args.input:
        input_dir = args.input
    else:
        input_dir = _latest_run(_CAMERAS_DIR)
        if input_dir is None:
            print(f"Нет run_* каталогов в {_CAMERAS_DIR}", file=sys.stderr)
            return 1
        print(f"Используется последний прогон: {input_dir.name}")

    if not input_dir.exists():
        print(f"Нет каталога: {input_dir}", file=sys.stderr)
        return 1

    out_dir = args.output or DEFAULT_OUTPUT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = collect_diffs(input_dir)
    total = sum(len(v) for v in cats.values())
    if total == 0:
        print(f"Нет diff-файлов в {input_dir}", file=sys.stderr)
        return 1

    print(f"Вход: {input_dir}")
    print(f"Вывод: {out_dir}")
    print(f"Всего diff-значений: {total}\n")

    report: dict = {"input": str(input_dir), "categories": {}}
    hist_lines: list[str] = []

    for cat, vals in cats.items():
        if not vals:
            continue
        arr = np.array(vals)
        report["categories"][cat] = {
            "count": len(vals),
            "min": round(float(arr.min()), 2),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "max": round(float(arr.max()), 2),
            "pct_above_30": round(float((arr > 30).mean() * 100), 1),
            "pct_above_40": round(float((arr > 40).mean() * 100), 1),
        }
        title = f"=== {cat} (n={len(vals)}) ==="
        print(title)
        hist = ascii_histogram(vals, bins=args.bins)
        print(hist)
        print()
        hist_lines += [title, hist, ""]

    # Все diff вместе
    all_vals = [v for vs in cats.values() for v in vs]
    if all_vals:
        arr_all = np.array(all_vals)
        report["all"] = {
            "count": len(all_vals),
            "mean": round(float(arr_all.mean()), 2),
            "median": round(float(np.median(arr_all)), 2),
            "p75": round(float(np.percentile(arr_all, 75)), 2),
            "p90": round(float(np.percentile(arr_all, 90)), 2),
            "p95": round(float(np.percentile(arr_all, 95)), 2),
            "pct_above_30": round(float((arr_all > 30).mean() * 100), 1),
            "pct_above_40": round(float((arr_all > 40).mean() * 100), 1),
            "suggested_threshold": _suggest_threshold(arr_all),
        }
        print(f"=== ВСЕ ВМЕСТЕ (n={len(all_vals)}) ===")
        print(ascii_histogram(all_vals, bins=args.bins))
        t = report["all"].get("suggested_threshold")
        if t:
            print(f"\n  Предложенный порог: {t} (отсекает нижние 20% срабатываний)")

    # Сохраняем
    json_path = out_dir / "diff_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path = out_dir / "hist_all.txt"
    hist_path.write_text("\n".join(hist_lines), encoding="utf-8")

    # PNG если matplotlib доступен
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        COLORS = {"low_diff": "#5599ff", "raw_diff": "#ff9944", "detected": "#44cc66"}
        THRESHOLDS = [(6, "green", "порог шума"), (30, "orange", "~p20"), (40, "red", "~p95")]

        # Создаём только нужные сабплоты (без пустых) + 1 суммарный
        nonempty = [(cat, vals) for cat, vals in cats.items() if vals]
        ncols = len(nonempty) + 1  # +1 суммарный
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        def _add_thresholds(ax):
            for thresh, color, lbl in THRESHOLDS:
                ax.axvline(thresh, color=color, linestyle="--", alpha=0.8,
                           linewidth=1.5, label=f"thr={thresh} ({lbl})")

        for ax, (cat, vals) in zip(axes, nonempty):
            color = COLORS.get(cat, "#aaaaaa")
            ax.hist(vals, bins=args.bins, edgecolor="white", linewidth=0.3, color=color, alpha=0.85)
            ax.set_title(f"{cat}\nn={len(vals)}, med={np.median(vals):.1f}", fontsize=9)
            ax.set_xlabel("diff")
            ax.set_ylabel("count")
            _add_thresholds(ax)
            ax.legend(fontsize=7)

        # Суммарный: все категории наложены друг на друга
        all_ax = axes[-1]
        for cat, vals in nonempty:
            color = COLORS.get(cat, "#aaaaaa")
            all_ax.hist(vals, bins=args.bins * 2, edgecolor="none", alpha=0.5, color=color, label=cat)
        if all_vals:
            all_ax.set_title(
                f"Все категории\nn={len(all_vals)}, med={np.median(all_vals):.1f}", fontsize=9
            )
            all_ax.set_xlabel("diff")
            _add_thresholds(all_ax)
            all_ax.legend(fontsize=7)

        plt.tight_layout()
        png_path = out_dir / "hist_all.png"
        plt.savefig(str(png_path), dpi=130)
        plt.close()
        print(f"\nPNG: {png_path}")
    except ImportError:
        pass  # matplotlib не обязателен

    print(f"JSON: {json_path}")
    print(f"TXT:  {hist_path}")
    return 0


def _print_runs_breakdown(input_dir: Path) -> None:
    """Печатает количество diff-файлов по каждому прогону."""
    runs = (
        [input_dir]
        if (input_dir / "raw").exists() or any(input_dir.glob("*.jpg"))
        else sorted(input_dir.iterdir())
    )
    has_breakdown = False
    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        low = hi = det = 0
        raw_dir = run_dir / "raw"
        if raw_dir.is_dir():
            for f in raw_dir.glob("*.jpg"):
                if "_low_diff" in f.name:
                    low += 1
                elif "_raw_diff" in f.name:
                    hi += 1
        for f in run_dir.glob("*.jpg"):
            if re.search(r"_low_p\d+\.jpg$", f.name) or re.search(r"_p\d+\.jpg$", f.name):
                det += 1
        if low + hi + det > 0:
            if not has_breakdown:
                print(f"\n  {'Прогон':35s}  low_diff  raw_diff  detected")
                has_breakdown = True
            print(f"  {run_dir.name:35s}  {low:8d}  {hi:8d}  {det:8d}")


def _suggest_threshold(arr: np.ndarray) -> float | None:
    """Предлагает порог как 20-й перцентиль (отсекает фоновый шум)."""
    if len(arr) < 5:
        return None
    return round(float(np.percentile(arr, 20)), 1)


if __name__ == "__main__":
    raise SystemExit(main())
