"""Сборка датасета групп v5 (multi-label) из ПРЕДЫДУЩЕГО датасета + СГЛАЖЕННЫХ меток инференса.

Обобщение build_groups_v4.py. Отличия:
  - источник-датасет читается через labels.json (multi-label формат v4: single/+multi/+labels.json),
    а не по каталогам-классам (как было для single-label v3);
  - источник-инференс — СГЛАЖЕННЫЕ метки (smoothed/<date>/labels.json, их ведёт разметчик 2_label_ui),
    а НЕ images/<date>/labels.json. Файлы кропов лежат в sibling images/<date>/<rel> (в smoothed/ кропов нет).

Источники по умолчанию (v5):
  --src-dataset   .data/groups/v4/dataset                       (multi-label labels.json)
  --src-inference .data/groups/v4/inference                     (→ smoothed/*/labels.json, файлы в images/)

Формат вывода (.data/groups/v5/dataset/), как у v4:
  single/<class>/img.jpg   — один класс
  multi/img.jpg            — несколько классов
  labels.json              — v2 {rel: [classes]}
  sources.csv, dataset.json

Прореживание резидентов: farthest-point по pixel-diff, окно WINDOW сек по (камера, дата), K на бакет.
Миноры (delivery/utilities/guest) и мульти — целиком.

Usage:
    python scripts/train/build_groups_v5.py --dry-run   # оценка (сколько кропов каких классов)
    python scripts/train/build_groups_v5.py             # собрать в .data/groups/v5/dataset
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

from common.utils.classes import GROUP_CLASSES  # noqa: E402
from common.utils import multilabel as ml       # noqa: E402

REAL = set(GROUP_CLASSES)
DEDUP_CLASS = "1_resident"
WINDOW_SEC = 30
K_PER_BUCKET = 5


def _parse_name(name: str):
    """cam_01_9_d_20260628_143742_559207_… → (cam, date, sec_of_day). None если не разобрать."""
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


def _read_labels_dir(labels_base: Path, resolve_base: Path, source: str, items: dict) -> int:
    """Читает labels.json из labels_base; ключ = rel-путь, файл кропа = resolve_base/<ключ>.
    Дедуп по basename (кропы глобально уникальны по таймстампу). Возвращает число добавленных.
    Фолбэк на каталоги-классы, если labels.json нет (single-label источник)."""
    added = 0
    lj = labels_base / "labels.json"
    if lj.is_file():
        try:
            raw = json.loads(lj.read_text(encoding="utf-8"))
        except Exception:
            return 0
        L = raw.get("labels", raw)
        for key, val in L.items():
            classes = _clean_classes(val)
            if not classes:
                continue
            f = resolve_base / key
            if not f.is_file():
                continue
            name = Path(key).name
            if name not in items:                # дедуп по basename: первый источник (датасет) в приоритете
                items[name] = {"path": f, "classes": classes, "source": source}
                added += 1
    else:                                        # фолбэк: single-label по каталогам-классам
        for cls in GROUP_CLASSES:
            d = labels_base / cls
            if not d.is_dir():
                continue
            for f in d.glob("*.jpg"):
                if f.name not in items:
                    items[f.name] = {"path": f, "classes": [cls], "source": source}
                    added += 1
    return added


def _collect(src_dataset: Path, src_inference: Path) -> dict:
    """{basename: {"path", "classes", "source"}}. Приоритет — датасет (курированный), потом инференс."""
    items: dict[str, dict] = {}
    ds_src = f"{src_dataset.parent.name}dataset"
    _read_labels_dir(src_dataset, src_dataset, ds_src, items)          # 1) предыдущий датасет (multi-label)

    inf_images = src_inference / "images"
    inf_src = f"{src_inference.parent.name}smoothed"
    for lj in sorted((src_inference / "smoothed").glob("*/labels.json")):
        date = lj.parent.name                                          # 2) СГЛАЖЕННЫЕ метки; файлы в images/<date>/
        _read_labels_dir(lj.parent, inf_images / date, inf_src, items)
    return items


def _load_small(path: Path, size=(48, 48)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, size).astype(np.float32) if img is not None else None


def _farthest_point(names: list[str], items: dict, k: int) -> list[str]:
    """K максимально разных по pixel-diff (farthest-point sampling). ≤k → все."""
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
        nxt = max(mind, key=mind.get)
        chosen.append(nxt)
        del mind[nxt]
        for n in list(mind):
            d = dist(n, nxt)
            if d < mind[n]:
                mind[n] = d
    return chosen


def _dedup_residents(items: dict, window: int, k: int) -> tuple[set[str], dict]:
    """Прореживает кропы класса DEDUP_CLASS (только single-label резиденты). (оставленные, статистика)."""
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
    kept: set[str] = set(unparsed)
    n_big = 0
    for _key, names in buckets.items():
        if len(names) <= k:
            kept.update(names)
        else:
            n_big += 1
            kept.update(_farthest_point(sorted(names), items, k))
    stats = {"residents_in": len(res_names), "residents_kept": len(kept),
             "buckets": len(buckets), "buckets_thinned": n_big}
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка датасета групп v5 (multi-label): датасет + сглаженные метки")
    ap.add_argument("--src-dataset", type=Path, default=REPO_ROOT / ".data/groups/v4/dataset",
                    help="предыдущий датасет (multi-label labels.json или каталоги-классы)")
    ap.add_argument("--src-inference", type=Path, default=REPO_ROOT / ".data/groups/v4/inference",
                    help="каталог инференса: берутся smoothed/*/labels.json, файлы из images/")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / ".data/groups/v5/dataset")
    ap.add_argument("--version", type=int, default=5)
    ap.add_argument("--window", type=int, default=WINDOW_SEC)
    ap.add_argument("--k", type=int, default=K_PER_BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="только оценка, без копирования")
    args = ap.parse_args()

    print(f"Сбор: датасет {args.src_dataset}  +  сглаженные метки {args.src_inference}/smoothed/…", flush=True)
    items = _collect(args.src_dataset, args.src_inference)
    by_source = defaultdict(int)
    for it in items.values():
        by_source[it["source"]] += 1
    print(f"  всего уникальных кропов: {len(items)}  ({dict(by_source)})")

    print(f"\nПрореживание {DEDUP_CLASS}: окно {args.window}с, K={args.k}, farthest-point по pixel-diff…", flush=True)
    res_kept, st = _dedup_residents(items, args.window, args.k)
    print(f"  {DEDUP_CLASS}: {st['residents_in']} → {st['residents_kept']}  "
          f"(бакетов {st['buckets']}, прорежено {st['buckets_thinned']})")

    final = {n: it for n, it in items.items() if it["classes"] != [DEDUP_CLASS] or n in res_kept}

    per_class = defaultdict(int)
    n_single = n_multi = 0
    for it in final.values():
        for c in it["classes"]:
            per_class[c] += 1
        if len(it["classes"]) == 1:
            n_single += 1
        else:
            n_multi += 1
    print(f"\nИтог v{args.version}: {len(final)} кропов  (single={n_single}, multi={n_multi})")
    for c in GROUP_CLASSES:
        print(f"  {c:14} {per_class[c]}")

    if args.dry_run:
        print("\n(dry-run: файлы не копировались)")
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list] = {}
    src_rows = []
    copied = 0
    for name, it in sorted(final.items()):
        classes = it["classes"]
        rel = (Path("single") / classes[0] / name) if len(classes) == 1 else (Path("multi") / name)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(it["path"], dst)
            copied += 1
        rel_posix = rel.as_posix()
        labels[rel_posix] = classes
        src_rows.append({"name": name, "classes": "|".join(classes),
                         "source": it["source"], "orig_path": str(it["path"]), "dst_rel": rel_posix})

    ml.save_labels(out / "labels.json", labels, task="classify")
    with open(out / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "classes", "source", "orig_path", "dst_rel"])
        w.writeheader()
        w.writerows(src_rows)

    (out / "dataset.json").write_text(json.dumps({
        "version": args.version, "multi_label": True, "classes": GROUP_CLASSES,
        "total": len(final), "single": n_single, "multi": n_multi,
        "counts": {c: per_class[c] for c in GROUP_CLASSES},
        "resident_dedup": {"window_sec": args.window, "k_per_bucket": args.k,
                           "method": "farthest-point pixel-diff", **st},
        "sources": [f"{args.src_dataset.parent.name}/dataset",
                    f"{args.src_inference.parent.name}/inference/smoothed"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out.parent / "_FORMAT_multi-label.txt").write_text(
        f"v{args.version} — multi-label: single/<class>/ + multi/ + labels.json (v2 {{img:[classes]}}).\n"
        f"Собран build_groups_v5.py из {args.src_dataset} + {args.src_inference}/smoothed; "
        f"резиденты прорежены (окно {args.window}с, K={args.k}, farthest-point pixel-diff).\n", encoding="utf-8")

    print(f"\nСкопировано: {copied} файлов → {out}")
    print(f"labels.json: {len(labels)} записей;  sources.csv: {len(src_rows)} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
