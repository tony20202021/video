"""Сборка датасета групп v4 (multi-label формат) из v3-датасета + v3-инференса.

Источники:
  - .data/groups/v3/dataset/<class>/            — курированный (уже включает v1+v2)
  - .data/groups/v3/inference/images/*/labels.json — новое (ключи = абсолютные пути к кропам)

Формат v4 (.data/groups/v4/dataset/):
  single/<class>/img.jpg   — кроп с ОДНИМ классом
  multi/img.jpg            — кроп с НЕСКОЛЬКИМИ классами (в этих данных их 0)
  labels.json              — v2 {rel: [classes]}, ключи относительно каталога датасета
  sources.csv              — аудит (откуда взят каждый кроп)
  dataset.json             — метаданные (multi_label=true, счётчики)

Прореживание резидентов (только Модель 1; Модель 2/identify не трогаем):
  окно WINDOW сек по (камера, дата), K кропов на бакет, отбор — farthest-point по PIXEL-diff
  (5 максимально разных по картинке кадров на визит, а не 5 одинаковых стоячих).
  Миноры (2_delivery/3_utilities/4_guest) и мульти — целиком.

Usage:
    python scripts/train/build_groups_v4.py --dry-run     # только оценка, без копирования
    python scripts/train/build_groups_v4.py               # собрать в .data/groups/v4/dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.classes import GROUP_CLASSES, EXTRA_DATASET_DIRS
from common.utils import multilabel as ml

REAL = set(GROUP_CLASSES)
PSEUDO = set(EXTRA_DATASET_DIRS)
DEDUP_CLASS = "1_resident"           # какой класс прореживаем
WINDOW_SEC = 30                      # окно бакета
K_PER_BUCKET = 5                     # кропов на бакет (по 5 максимально разных)

_RE_DIFF = re.compile(r"diff([0-9]+\.?[0-9]*)")


def _parse_name(name: str):
    """cam_01_9_d_20260628_143742_559207_...  → (cam, date, sec_of_day). None если не разобрать."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if (len(p) == 8 and p.isdigit() and i + 1 < len(parts)
                and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
            t = parts[i + 1]
            return "_".join(parts[:i]), p, int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return None


def _clean_classes(vals) -> list[str]:
    """Настоящие классы (псевдо-метки skip/unknown/uncertain/new убираем)."""
    return sorted({c for c in ml.normalize_label(vals) if c in REAL})


def _collect() -> dict[str, dict]:
    """Собирает {name: {"path": Path, "classes": [...], "source": str}} из обоих источников.
    Дедуп по имени файла (кропы глобально уникальны по таймстампу)."""
    items: dict[str, dict] = {}

    # 1) v3 датасет — файлы по каталогам-классам
    ds = REPO_ROOT / ".data" / "groups" / "v3" / "dataset"
    for cls in GROUP_CLASSES:
        d = ds / cls
        if not d.is_dir():
            continue
        for f in d.glob("*.jpg"):
            items.setdefault(f.name, {"path": f, "classes": [cls], "source": "v3dataset"})

    # 2) v3 инференс — labels.json (ключи = абсолютные пути)
    inf = REPO_ROOT / ".data" / "groups" / "v3" / "inference" / "images"
    for lj in inf.glob("*/labels.json"):
        try:
            raw = json.loads(lj.read_text(encoding="utf-8"))
        except Exception:
            continue
        L = raw.get("labels", raw)
        for key, val in L.items():
            classes = _clean_classes(val)
            if not classes:
                continue
            p = Path(key)
            if not p.is_absolute():
                p = REPO_ROOT / key
            if not p.is_file():
                continue
            items.setdefault(p.name, {"path": p, "classes": classes, "source": "v3inference"})
    return items


def _load_small(path: Path, size=(48, 48)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size).astype(np.float32)


def _farthest_point(names: list[str], items: dict, k: int) -> list[str]:
    """K максимально разных по pixel-diff из names (farthest-point sampling). ≤k → все."""
    if len(names) <= k:
        return names
    arrs = {n: _load_small(items[n]["path"]) for n in names}
    valid = [n for n in names if arrs[n] is not None]
    if len(valid) <= k:
        return valid if valid else names[:k]

    def dist(a, b):
        return float(np.mean(np.abs(arrs[a] - arrs[b])))

    chosen = [valid[0]]
    mind = {n: dist(n, valid[0]) for n in valid if n != valid[0]}
    while len(chosen) < k and mind:
        nxt = max(mind, key=mind.get)      # самый далёкий от уже выбранных
        chosen.append(nxt)
        del mind[nxt]
        for n in list(mind):
            d = dist(n, nxt)
            if d < mind[n]:
                mind[n] = d
    return chosen


def _dedup_residents(items: dict, window: int, k: int) -> tuple[set[str], dict]:
    """Прореживает кропы класса DEDUP_CLASS. Возвращает (оставленные имена, статистика)."""
    res_names = [n for n, it in items.items() if it["classes"] == [DEDUP_CLASS]]
    buckets: dict[tuple, list[str]] = defaultdict(list)
    unparsed = []
    for n in res_names:
        pr = _parse_name(n)
        if pr is None:
            unparsed.append(n)
            continue
        cam, date, sod = pr
        buckets[(cam, date, sod // window)].append(n)

    kept: set[str] = set(unparsed)         # не разобрали имя — оставляем на всякий
    n_buckets_big = 0
    for key, names in buckets.items():
        if len(names) <= k:
            kept.update(names)
        else:
            n_buckets_big += 1
            kept.update(_farthest_point(sorted(names), items, k))
    stats = {"residents_in": len(res_names), "residents_kept": len(kept),
             "buckets": len(buckets), "buckets_thinned": n_buckets_big}
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка датасета групп v4 (multi-label)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / ".data" / "groups" / "v4" / "dataset")
    ap.add_argument("--window", type=int, default=WINDOW_SEC)
    ap.add_argument("--k", type=int, default=K_PER_BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="Только оценка, без копирования")
    args = ap.parse_args()

    print("Сбор источников (v3 датасет + v3 инференс)…", flush=True)
    items = _collect()
    by_class_all = defaultdict(int)
    for it in items.values():
        by_class_all["+".join(it["classes"])] += 1
    print(f"  всего уникальных кропов: {len(items)}")

    print(f"\nПрореживание {DEDUP_CLASS}: окно {args.window}с, K={args.k}, farthest-point по pixel-diff…",
          flush=True)
    res_kept, st = _dedup_residents(items, args.window, args.k)
    print(f"  {DEDUP_CLASS}: {st['residents_in']} → {st['residents_kept']}  "
          f"(бакетов {st['buckets']}, прорежено {st['buckets_thinned']})")

    # Финальный набор: не-резиденты целиком + оставленные резиденты
    final = {n: it for n, it in items.items()
             if it["classes"] != [DEDUP_CLASS] or n in res_kept}

    # Счётчики по классам (кроп считается в каждом своём классе)
    per_class = defaultdict(int)
    n_single = n_multi = 0
    for it in final.values():
        for c in it["classes"]:
            per_class[c] += 1
        if len(it["classes"]) == 1:
            n_single += 1
        else:
            n_multi += 1
    print(f"\nИтог v4: {len(final)} кропов  (single={n_single}, multi={n_multi})")
    for c in GROUP_CLASSES:
        print(f"  {c:14} {per_class[c]}")

    if args.dry_run:
        print("\n(dry-run: файлы не копировались)")
        return 0

    # ── Запись v4 ──────────────────────────────────────────────────────────────
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list] = {}
    src_rows = []
    copied = 0
    for name, it in sorted(final.items()):
        classes = it["classes"]
        if len(classes) == 1:
            rel = Path("single") / classes[0] / name
        else:
            rel = Path("multi") / name
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(it["path"], dst)
            copied += 1
        rel_posix = rel.as_posix()
        labels[rel_posix] = classes
        src_rows.append({"name": name, "classes": "|".join(classes),
                         "source": it["source"], "orig_path": str(it["path"]),
                         "dst_rel": rel_posix})

    ml.save_labels(out / "labels.json", labels, task="classify")

    with open(out / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "classes", "source", "orig_path", "dst_rel"])
        w.writeheader()
        w.writerows(src_rows)

    (out / "dataset.json").write_text(json.dumps({
        "version": 4, "multi_label": True, "classes": GROUP_CLASSES,
        "total": len(final), "single": n_single, "multi": n_multi,
        "counts": {c: per_class[c] for c in GROUP_CLASSES},
        "resident_dedup": {"window_sec": args.window, "k_per_bucket": args.k,
                           "method": "farthest-point pixel-diff", **st},
        "sources": ["v3/dataset", "v3/inference"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out.parent / "_FORMAT_multi-label.txt").write_text(
        "v4 — multi-label формат: single/<class>/ + multi/ + labels.json (v2 {img:[classes]}).\n"
        "Собран build_groups_v4.py из v3/dataset + v3/inference; резиденты прорежены "
        f"(окно {args.window}с, K={args.k}, farthest-point pixel-diff).\n", encoding="utf-8")

    print(f"\nСкопировано: {copied} файлов → {out}")
    print(f"labels.json: {len(labels)} записей")
    print(f"sources.csv: {len(src_rows)} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
