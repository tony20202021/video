"""Анализ: сравнение распределений вероятностей на одиночных vs мультиперсонных кропах.

Гипотеза: если в кропе несколько людей — модель даёт высокие вероятности
сразу по нескольким классам (несколько классов выше порога идентификации).

Входные данные:
  - identifications.csv из каждой даты (содержит p_<class> для каждого кропа)
  - YOLO на самих кропах — определяет сколько людей в конкретном кропе

Алгоритм:
  1. Загружает все identifications.csv
  2. На каждом кропе запускает YOLO (или берёт кэш из yolo_per_crop.json)
  3. Делит на 2 сегмента: single (YOLO=1) и multi (YOLO≥2)
  4. Для каждого сегмента:
     - среднее/медиана вероятности по каждому классу
     - гистограмма: сколько классов выше порога на одном кропе
     - scatter: top-1 prob vs top-2 prob
  5. Сохраняет JSON + PNG в analysis/

Usage:
  python scripts/analysis/3_multi_prob_analysis.py
  python scripts/analysis/3_multi_prob_analysis.py --threshold 0.70 --sample 300
  python scripts/analysis/3_multi_prob_analysis.py --use-cache   # взять YOLO из кэша
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INFERENCE_ROOT = Path(".data/residents/v1/inference/images")
MODEL_PATH = Path(".models/detect/yolov8n.onnx")
DEFAULT_OUT = Path(".data/residents/v1/analysis")
CACHE_FILE = DEFAULT_OUT / "yolo_per_crop.json"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from common.utils.person_detector import load_model, detect_people


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", type=Path, default=INFERENCE_ROOT)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Порог идентификации")
    ap.add_argument("--yolo-conf", type=float, default=0.25,
                    help="Порог детекции YOLO")
    ap.add_argument("--sample", type=int, default=0,
                    help="Случайная выборка N кропов (0 = все)")
    ap.add_argument("--use-cache", action="store_true",
                    help="Использовать кэш YOLO (yolo_per_crop.json)")
    return ap.parse_args()


# ─── YOLO ────────────────────────────────────────────────────────────────────


# ─── CSV loading ─────────────────────────────────────────────────────────────

def load_all_csv(inference_root: Path) -> list[dict]:
    rows = []
    for date_dir in sorted(inference_root.iterdir()):
        if not date_dir.is_dir():
            continue
        csv_path = date_dir / "identifications.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # восстанавливаем путь к файлу кропа
                out_class = row.get("out_class", "unknown_resident") or "unknown_resident"
                crop_path = date_dir / out_class / row["crop"]
                if crop_path.exists():
                    row["_path"] = str(crop_path)
                    row["_date"] = date_dir.name
                    rows.append(row)
    return rows


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyse_segments(rows: list[dict], threshold: float) -> dict:
    """Разбивает строки на single/multi, считает статистику вероятностей."""
    prob_cols = sorted(k for k in rows[0] if k.startswith("p_"))
    classes = [c[2:] for c in prob_cols]  # strip "p_"

    segments = {"single": [], "multi": [], "zero": []}
    for r in rows:
        n = r.get("_yolo_persons", -1)
        if n == 0:
            segments["zero"].append(r)
        elif n == 1:
            segments["single"].append(r)
        else:
            segments["multi"].append(r)

    result = {}
    for seg_name, seg_rows in segments.items():
        if not seg_rows:
            result[seg_name] = {"n": 0}
            continue

        probs_matrix = np.array(
            [[float(r.get(c, 0)) for c in prob_cols] for r in seg_rows]
        )  # (N, n_classes)

        top1 = probs_matrix.max(axis=1)
        top2 = np.sort(probs_matrix, axis=1)[:, -2]
        classes_above = (probs_matrix >= threshold).sum(axis=1)

        result[seg_name] = {
            "n": len(seg_rows),
            "classes": classes,
            "mean_prob": probs_matrix.mean(axis=0).round(4).tolist(),
            "median_prob": np.median(probs_matrix, axis=0).round(4).tolist(),
            "top1_mean": float(top1.mean().round(4)),
            "top1_median": float(np.median(top1).round(4)),
            "top2_mean": float(top2.mean().round(4)),
            "top2_median": float(np.median(top2).round(4)),
            "classes_above_threshold": {
                "threshold": threshold,
                "distribution": dict(Counter(classes_above.tolist())),
                "mean": float(classes_above.mean().round(3)),
            },
            "_top1": top1.tolist(),
            "_top2": top2.tolist(),
            "_classes_above": classes_above.tolist(),
            "_probs_matrix": probs_matrix.tolist(),
        }

    return result


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_results(analysis: dict, out_dir: Path, threshold: float) -> None:
    classes = analysis["single"].get("classes") or analysis["multi"].get("classes", [])
    n_classes = len(classes)
    segs = {k: v for k, v in analysis.items() if v.get("n", 0) > 0 and "_top1" in v}

    colors = {"single": "#4c9be8", "multi": "#e84c4c", "zero": "#aaaaaa"}

    # ── 1. Среднее P по классам ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.8), 5))
    x = np.arange(n_classes)
    width = 0.35
    offsets = np.linspace(-(len(segs)-1)*width/2, (len(segs)-1)*width/2, len(segs))
    for i, (seg_name, seg) in enumerate(segs.items()):
        ax.bar(x + offsets[i], seg["mean_prob"], width,
               label=f"{seg_name} (n={seg['n']})", color=colors.get(seg_name, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_ylabel("Средняя вероятность")
    ax.set_title("Среднее P по классам: single vs multi-person кропы")
    ax.legend()
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label=f"threshold={threshold}")
    fig.tight_layout()
    fig.savefig(out_dir / "prob_by_class.png", dpi=120)
    plt.close(fig)

    # ── 2. Гистограмма: кол-во классов выше порога ───────────────────────────
    fig, axes = plt.subplots(1, len(segs), figsize=(5 * len(segs), 4), sharey=False)
    if len(segs) == 1:
        axes = [axes]
    for ax, (seg_name, seg) in zip(axes, segs.items()):
        above = seg["_classes_above"]
        counts = Counter(above)
        xs = sorted(counts)
        ax.bar([str(x) for x in xs], [counts[x] for x in xs],
               color=colors.get(seg_name, "#888"))
        ax.set_title(f"{seg_name}  (n={seg['n']})\nсреднее: {seg['classes_above_threshold']['mean']:.2f} классов")
        ax.set_xlabel("Классов выше порога")
        ax.set_ylabel("Кропов")
    fig.suptitle(f"Кол-во классов ≥ {threshold} на один кроп", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "classes_above_threshold.png", dpi=120)
    plt.close(fig)

    # ── 3. Scatter: top-1 prob vs top-2 prob ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    for seg_name, seg in segs.items():
        ax.scatter(seg["_top1"], seg["_top2"],
                   alpha=0.25, s=8, label=f"{seg_name} (n={seg['n']})",
                   color=colors.get(seg_name, "#888"))
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Top-1 вероятность")
    ax.set_ylabel("Top-2 вероятность")
    ax.set_title("Top-1 vs Top-2 вероятность по кропам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top1_vs_top2.png", dpi=120)
    plt.close(fig)

    print(f"  Графики → {out_dir}/prob_by_class.png, classes_above_threshold.png, top1_vs_top2.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Загрузка identifications.csv…")
    rows = load_all_csv(args.inference)
    print(f"  Записей: {len(rows)}")

    if not rows:
        print("Нет данных.")
        return

    import random
    if args.sample and len(rows) > args.sample:
        rows = random.sample(rows, args.sample)
        print(f"  Выборка: {len(rows)} кропов")

    # ── YOLO per-crop ─────────────────────────────────────────────────────────
    cache: dict[str, int] = {}
    if args.use_cache and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"  Кэш YOLO: {len(cache)} записей")

    paths_need_yolo = [r["_path"] for r in rows if r["_path"] not in cache]
    if paths_need_yolo:
        print(f"Запуск YOLO на {len(paths_need_yolo)} кропах (conf={args.yolo_conf})…")
        sess = load_model(args.model)
        if sess is None:
            return
        t0 = time.time()
        for i, p in enumerate(paths_need_yolo):
            bgr = cv2.imread(p)
            cache[p] = len(detect_people(sess, bgr, conf_threshold=args.yolo_conf, nms_threshold=0.45)) if bgr is not None else 0
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(paths_need_yolo)}  ({time.time()-t0:.0f}s)")
        # сохраняем кэш
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
        print(f"  Кэш сохранён → {CACHE_FILE}  ({time.time()-t0:.0f}s)")
    else:
        print("  Все кропы в кэше — YOLO не запускается")

    for r in rows:
        r["_yolo_persons"] = cache.get(r["_path"], -1)

    # ── Анализ ────────────────────────────────────────────────────────────────
    analysis = analyse_segments(rows, args.threshold)

    print(f"\n{'═'*55}")
    print(f"  Порог идентификации: {args.threshold}")
    for seg_name, seg in analysis.items():
        if not seg.get("n"):
            continue
        cab = seg.get("classes_above_threshold", {})
        dist = cab.get("distribution", {})
        print(f"\n  ── {seg_name.upper()}  (n={seg['n']}) ──")
        print(f"  Top-1 prob:  mean={seg['top1_mean']:.3f}  median={seg['top1_median']:.3f}")
        print(f"  Top-2 prob:  mean={seg['top2_mean']:.3f}  median={seg['top2_median']:.3f}")
        print(f"  Классов ≥ {args.threshold}:  среднее={cab.get('mean', 0):.2f}")
        print(f"  Распределение (N классов ≥ порога): {dict(sorted(dist.items()))}")

    # убираем матрицы из JSON (слишком большие), оставляем только статистику
    for seg in analysis.values():
        seg.pop("_top1", None)
        seg.pop("_top2", None)
        seg.pop("_classes_above", None)
        seg.pop("_probs_matrix", None)

    out_json = args.out / "multi_prob_analysis.json"
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    print(f"\n  JSON → {out_json}")

    # ── Графики ──────────────────────────────────────────────────────────────
    # нужно заново посчитать для графиков — перегружаем
    rows2 = load_all_csv(args.inference)
    if args.sample and len(rows2) > args.sample:
        import random as _r
        _r.seed(42)
        rows2 = _r.sample(rows2, args.sample)
    for r in rows2:
        r["_yolo_persons"] = cache.get(r["_path"], -1)
    analysis2 = analyse_segments(rows2, args.threshold)
    plot_results(analysis2, args.out, args.threshold)


if __name__ == "__main__":
    main()
