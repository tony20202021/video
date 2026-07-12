"""Медленный анализ: YOLO на самих кропах — сколько людей внутри одного кропа.

Запускает YOLOv8n (с NMS) на каждом jpg из inference/images/<date>/
и считает, сколько bounding box'ов класса 'person' детектируется внутри кропа.

Usage:
  python scripts/analysis/2_crop_occupancy_yolo.py
  python scripts/analysis/2_crop_occupancy_yolo.py --conf 0.35 --sample 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from common.utils.person_detector import load_model, detect_people

INFERENCE_ROOT = Path(".data/residents/v1/inference/images")
MODEL_PATH = Path(".models/detect/yolov8n.onnx")
DEFAULT_OUT = Path(".data/residents/v1/analysis")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", type=Path, default=INFERENCE_ROOT)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--sample", type=int, default=0,
                    help="Случайная выборка N кропов на дату (0 = все)")
    return ap.parse_args()


def analyse_date(date_dir: Path, sess, conf: float, nms: float, sample: int) -> dict:
    jpgs = sorted(date_dir.rglob("*.jpg"))
    if sample and len(jpgs) > sample:
        jpgs = random.sample(jpgs, sample)

    person_counts: Counter[int] = Counter()
    t0 = time.time()

    for i, jpg in enumerate(jpgs):
        bgr = cv2.imread(str(jpg))
        if bgr is None:
            continue
        n = len(detect_people(sess, bgr, conf_threshold=conf, nms_threshold=nms))
        person_counts[n] += 1
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(jpgs)}  ({time.time()-t0:.0f}s)")

    total = sum(person_counts.values())
    zero = person_counts[0]
    single = person_counts[1]
    multi = total - zero - single

    pct = lambda n: round(100 * n / total, 1) if total else 0
    return {
        "date": date_dir.name,
        "sampled": len(jpgs),
        "conf_thresh": conf,
        "nms_thresh": nms,
        "zero_persons": zero,   "zero_pct": pct(zero),
        "single_person": single, "single_pct": pct(single),
        "multi_person": multi,   "multi_pct": pct(multi),
        "distribution": dict(sorted(person_counts.items())),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.model.is_file():
        print(f"Модель не найдена: {args.model}")
        return

    print(f"Загрузка {args.model}…")
    sess = load_model(args.model)
    if sess is None:
        return
    print(f"  conf={args.conf}  nms={args.nms}  sample={args.sample or 'все'}")

    date_dirs = sorted(d for d in args.inference.iterdir() if d.is_dir())
    if not date_dirs:
        print(f"Нет дат в {args.inference}")
        return

    results = []
    for d in date_dirs:
        print(f"\n{'─'*50}")
        print(f"  {d.name}…")
        r = analyse_date(d, sess, args.conf, args.nms, args.sample)
        results.append(r)
        print(f"  Кропов проверено:  {r['sampled']}")
        print(f"  0 человек:         {r['zero_persons']}  ({r['zero_pct']}%)")
        print(f"  1 человек:         {r['single_person']}  ({r['single_pct']}%)")
        print(f"  2+ человек:        {r['multi_person']}  ({r['multi_pct']}%)")
        print(f"  Распределение:     {r['distribution']}")

    total_sampled = sum(r["sampled"] for r in results)
    total_zero   = sum(r["zero_persons"] for r in results)
    total_single = sum(r["single_person"] for r in results)
    total_multi  = sum(r["multi_person"] for r in results)
    dist_total: Counter[int] = Counter()
    for r in results:
        for k, v in r["distribution"].items():
            dist_total[int(k)] += v

    pct = lambda n: round(100 * n / total_sampled, 1) if total_sampled else 0
    summary = {
        "dates": [r["date"] for r in results],
        "model": str(args.model),
        "conf_thresh": args.conf,
        "nms_thresh": args.nms,
        "total_sampled": total_sampled,
        "zero_persons": total_zero,   "zero_pct": pct(total_zero),
        "single_person": total_single, "single_pct": pct(total_single),
        "multi_person": total_multi,   "multi_pct": pct(total_multi),
        "distribution": dict(sorted(dist_total.items())),
        "per_date": results,
    }

    print(f"\n{'═'*50}")
    print(f"  ИТОГО ({len(results)} дат, {total_sampled} кропов)")
    print(f"  0 человек:   {total_zero}  ({summary['zero_pct']}%)")
    print(f"  1 человек:   {total_single}  ({summary['single_pct']}%)")
    print(f"  2+ человек:  {total_multi}  ({summary['multi_pct']}%)")
    print(f"  Распределение: {dict(sorted(dist_total.items()))}")

    out_path = args.out / "occupancy_yolo.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
