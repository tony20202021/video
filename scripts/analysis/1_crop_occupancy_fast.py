"""Быстрый анализ заполняемости кропов по именам файлов.

Парсит pNofM из имён файлов:
  p1of1 → в кадре был 1 человек (одиночный кроп)
  p2of3 → в кадре было 3 человека (мультиперсонный кадр)

Выводит:
  - по дате: сколько кадров одиночных / мультиперсонных
  - общая статистика
  - распределение по M (кол-во людей в кадре)

Usage:
  python scripts/analysis/1_crop_occupancy_fast.py
  python scripts/analysis/1_crop_occupancy_fast.py --out .data/residents/v1/analysis
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RE_POF = re.compile(r"_p(\d+)of(\d+)_")

INFERENCE_ROOT = Path(".data/residents/v1/inference/images")
DEFAULT_OUT = Path(".data/residents/v1/analysis")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", type=Path, default=INFERENCE_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def analyse_date(date_dir: Path) -> dict:
    """Анализирует одну дату. Считает уникальные кадры (по timestamp) и их M."""
    # timestamp = часть имени до _pNofM, идентифицирует один момент во времени
    frame_m: dict[str, int] = {}  # frame_key → M (кол-во людей в кадре)

    for jpg in date_dir.rglob("*.jpg"):
        m = RE_POF.search(jpg.name)
        if not m:
            continue
        n_person = int(m.group(1))
        total = int(m.group(2))
        # frame_key = всё до _pN (без per-person части)
        frame_key = RE_POF.sub("_FRAME_", jpg.name)
        # берём max на случай несоответствия (должно быть одинаково)
        frame_m[frame_key] = max(frame_m.get(frame_key, 0), total)

    m_counter: Counter[int] = Counter(frame_m.values())
    total_frames = sum(m_counter.values())
    single = m_counter[1]
    multi = total_frames - single

    return {
        "date": date_dir.name,
        "total_frames": total_frames,
        "single_person_frames": single,
        "multi_person_frames": multi,
        "single_pct": round(100 * single / total_frames, 1) if total_frames else 0,
        "multi_pct": round(100 * multi / total_frames, 1) if total_frames else 0,
        "m_distribution": dict(sorted(m_counter.items())),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted(d for d in args.inference.iterdir() if d.is_dir())
    if not date_dirs:
        print(f"Нет дат в {args.inference}")
        return

    results = []
    for d in date_dirs:
        r = analyse_date(d)
        results.append(r)
        print(f"\n{'─'*50}")
        print(f"  Дата:              {r['date']}")
        print(f"  Кадров всего:      {r['total_frames']}")
        print(f"  Одиночных (M=1):   {r['single_person_frames']}  ({r['single_pct']}%)")
        print(f"  Мультиперсонных:   {r['multi_person_frames']}  ({r['multi_pct']}%)")
        print(f"  Распределение M:   {r['m_distribution']}")

    # Сводная
    total_f = sum(r["total_frames"] for r in results)
    total_s = sum(r["single_person_frames"] for r in results)
    total_m = sum(r["multi_person_frames"] for r in results)
    m_total: Counter[int] = Counter()
    for r in results:
        for k, v in r["m_distribution"].items():
            m_total[int(k)] += v

    summary = {
        "dates": [r["date"] for r in results],
        "total_frames": total_f,
        "single_person_frames": total_s,
        "multi_person_frames": total_m,
        "single_pct": round(100 * total_s / total_f, 1) if total_f else 0,
        "multi_pct": round(100 * total_m / total_f, 1) if total_f else 0,
        "m_distribution": dict(sorted(m_total.items())),
        "per_date": results,
    }

    print(f"\n{'═'*50}")
    print(f"  ИТОГО ({len(results)} дат)")
    print(f"  Кадров:           {total_f}")
    print(f"  Одиночных (M=1):  {total_s}  ({summary['single_pct']}%)")
    print(f"  Мультиперсонных:  {total_m}  ({summary['multi_pct']}%)")
    print(f"  Распределение M:  {dict(sorted(m_total.items()))}")

    out_path = args.out / "occupancy_fast.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
