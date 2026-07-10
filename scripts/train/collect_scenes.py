"""Сбор кропов жителей по сценам для полу-автоматической разметки.

Отличие от collect_residents.py:
  - НЕ применяет diff-фильтр (чтобы intra-сцена была мала и хорошо отделялась)
  - Разбивает поток кадров на сцены по временному разрыву и/или pixel-diff
  - Выбирает представительный кадр (наиболее резкий) для ручной разметки
  - Сохраняет scenes.json с разбивкой на сцены

Workflow:
  1. Запустить collect_scenes.py → scene_pool/ + scenes.json + representatives/
  2. Размечать только representatives/ (по 1 кадру на сцену)
  3. Запустить propagate_scenes.py → авторазметка остальных кадров в сцене
  4. При необходимости повторить 2–3 для пограничных сцен
  5. Запустить filter_residents.py --min-diff N → финальный отбор разнообразных кадров

Usage:
    python scripts/train/collect_scenes.py
    python scripts/train/collect_scenes.py --out .data/residents/v0/scene_pool
    python scripts/train/collect_scenes.py --scene-gap 120 --scene-diff 50
    python scripts/train/collect_scenes.py --min-blur 40 --min-conf 0.35
    python scripts/train/collect_scenes.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASSES = ["1_resident", "4_guest"]

_RE_FRAME = re.compile(
    r"^(?P<cam>cam_[^_]+)_\d+_[a-z]_(?P<date>\d{8})_(?P<time>\d{6})_\d+_.*_conf(?P<conf>\d+\.\d+)",
    re.IGNORECASE,
)
_RE_PERSONS = re.compile(r"_p\d+of(?P<total>\d+)_", re.IGNORECASE)


def _parse_frame(name: str) -> tuple[str, str, int, float] | None:
    m = _RE_FRAME.match(name)
    if not m:
        return None
    t = m.group("time")
    sod = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return m.group("cam"), m.group("date"), sod, float(m.group("conf"))


def _persons_in_frame(name: str) -> int:
    m = _RE_PERSONS.search(name)
    return int(m.group("total")) if m else 1


def _load_labels(date_dir: Path) -> dict[str, str]:
    lf = date_dir / "labels.json"
    if not lf.exists():
        return {}
    try:
        data = json.loads(lf.read_text())
        raw = data.get("labels", data)
        return {Path(k).name: v for k, v in raw.items() if isinstance(v, str)}
    except Exception:
        return {}


def _blur_score(path: Path) -> float:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_diff(path_a: Path, path_b: Path) -> float:
    a = cv2.imread(str(path_a))
    b = cv2.imread(str(path_b))
    if a is None or b is None:
        return 0.0
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def collect_scenes(
    sources: list[Path],
    out_dir: Path,
    *,
    collect_classes: set[str] | None = None,
    max_persons: int = 1,
    min_conf: float = 0.0,
    min_blur: float = 0.0,
    interval: int = 2,
    scene_gap_sec: int = 120,
    scene_diff: float = 50.0,
    dry_run: bool = False,
) -> dict:
    """Возвращает статистику: {scenes, frames, representatives}."""
    if collect_classes is None:
        collect_classes = set(DEFAULT_CLASSES)

    out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = out_dir / "representatives"
    if not dry_run:
        reps_dir.mkdir(exist_ok=True)

    seen_names: set[str] = {f.name for f in out_dir.iterdir()
                             if f.is_file() and f.suffix.lower() in IMAGE_EXTS}

    # ── Сбор кандидатов (без diff-фильтра) ──────────────────────────────────

    # Кэш labels.json
    _labels_cache: dict[Path, dict[str, str]] = {}
    def _get_labels(date_dir: Path) -> dict[str, str]:
        if date_dir not in _labels_cache:
            _labels_cache[date_dir] = _load_labels(date_dir)
        return _labels_cache[date_dir]

    # Временна́я дедупликация (интервал, без conf_delta — берём лучший)
    temporal: dict[tuple, tuple[float, Path]] = {}
    skipped_labels = skipped_persons = skipped_conf = 0

    for src_dir in sources:
        if not src_dir.is_dir():
            print(f"  [!] Не найдено: {src_dir}", file=sys.stderr)
            continue
        labels = _get_labels(src_dir.parent)

        for f in sorted(f for f in src_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS):
            if f.name in seen_names:
                continue
            if f.name in labels and labels[f.name] not in collect_classes:
                skipped_labels += 1
                continue
            if max_persons > 0 and _persons_in_frame(f.stem) > max_persons:
                skipped_persons += 1
                continue
            parsed = _parse_frame(f.stem)
            if parsed is None:
                temporal[("__noparse__", f.name)] = (0.0, f)
                continue
            cam, date, sod, conf = parsed
            if min_conf > 0 and conf < min_conf:
                skipped_conf += 1
                continue
            if interval > 0:
                key = (cam, date, sod // interval)
                if conf > temporal.get(key, (-1.0, None))[0]:
                    temporal[key] = (conf, f)
            else:
                temporal[(cam, date, sod)] = (conf, f)

        # Также сканируем соседние классы через labels.json
        date_dir = src_dir.parent
        for sibling in sorted(date_dir.iterdir()):
            if not sibling.is_dir() or sibling.name == src_dir.name:
                continue
            for f in sorted(f for f in sibling.iterdir()
                            if f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                if f.name in seen_names:
                    continue
                eff = _get_labels(date_dir).get(f.name)
                if eff not in collect_classes:
                    continue
                if max_persons > 0 and _persons_in_frame(f.stem) > max_persons:
                    skipped_persons += 1
                    continue
                parsed = _parse_frame(f.stem)
                if parsed is None:
                    continue
                cam, date, sod, conf = parsed
                if min_conf > 0 and conf < min_conf:
                    skipped_conf += 1
                    continue
                if interval > 0:
                    key = (cam, date, sod // interval)
                    if conf > temporal.get(key, (-1.0, None))[0]:
                        temporal[key] = (conf, f)
                else:
                    temporal[(cam, date, sod)] = (conf, f)

    candidates = [f for _, f in temporal.values()]

    # ── Blur-фильтр ─────────────────────────────────────────────────────────
    skipped_blur = 0
    if min_blur > 0:
        passed = []
        for f in candidates:
            b = _blur_score(f)
            if b >= min_blur:
                passed.append(f)
            else:
                skipped_blur += 1
        candidates = passed

    print(f"Кандидатов после фильтров: {len(candidates)}")
    print(f"  Пропущено (labels):   {skipped_labels}")
    print(f"  Пропущено (persons):  {skipped_persons}")
    print(f"  Пропущено (conf):     {skipped_conf}")
    print(f"  Пропущено (blur):     {skipped_blur}")

    # ── Разбивка на сцены ────────────────────────────────────────────────────
    # Группируем по (cam, date), сортируем по sod
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    noparse_files: list[Path] = []
    for f in candidates:
        p = _parse_frame(f.stem)
        if p is None:
            noparse_files.append(f)
            continue
        cam, date, sod, _ = p
        groups[(cam, date)].append((sod, f))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    scenes: list[dict] = []
    scene_id = 0

    for (cam, date), items in sorted(groups.items()):
        current: list[Path] = []
        prev_path: Path | None = None
        prev_sod: int | None = None

        for sod, path in items:
            if prev_sod is None:
                current.append(path)
                prev_sod = sod
                prev_path = path
                continue

            gap = sod - prev_sod
            boundary = False
            reason = ""

            if gap > scene_gap_sec:
                boundary = True
                reason = f"gap={gap}s"
            elif scene_diff > 0 and prev_path is not None:
                d = _mean_diff(prev_path, path)
                if d > scene_diff:
                    boundary = True
                    reason = f"diff={d:.0f}"

            if boundary:
                if current:
                    scenes.append({
                        "scene_id": scene_id,
                        "cam": cam,
                        "date": date,
                        "start_time": _parse_frame(current[0].stem)[2] if _parse_frame(current[0].stem) else 0,
                        "frames": [f.name for f in current],
                        "split_reason": reason,
                    })
                    scene_id += 1
                current = [path]
            else:
                current.append(path)

            prev_sod = sod
            prev_path = path

        if current:
            scenes.append({
                "scene_id": scene_id,
                "cam": cam,
                "date": date,
                "start_time": _parse_frame(current[0].stem)[2] if _parse_frame(current[0].stem) else 0,
                "frames": [f.name for f in current],
                "split_reason": "",
            })
            scene_id += 1

    # Кадры без парсинга — каждый отдельная сцена
    for f in noparse_files:
        scenes.append({
            "scene_id": scene_id,
            "cam": "unknown",
            "date": "unknown",
            "start_time": 0,
            "frames": [f.name],
            "split_reason": "noparse",
        })
        scene_id += 1

    print(f"\nСцен:   {len(scenes)}")
    lens = [len(s["frames"]) for s in scenes]
    if lens:
        print(f"  кадров/сцена: min={min(lens)}  median={sorted(lens)[len(lens)//2]}  max={max(lens)}")

    # ── Выбор представителя (наиболее резкий кадр) ──────────────────────────
    # Строим lookup имя → источник
    name_to_src: dict[str, Path] = {}
    for f in candidates:
        name_to_src[f.name] = f
    for f in noparse_files:
        name_to_src[f.name] = f

    for scene in scenes:
        best_name = scene["frames"][0]
        best_blur = -1.0
        for name in scene["frames"]:
            src = name_to_src.get(name)
            if src is None:
                continue
            b = _blur_score(src)
            if b > best_blur:
                best_blur = b
                best_name = name
        scene["representative"] = best_name
        scene["representative_blur"] = round(best_blur, 1)

    # ── Копирование файлов ───────────────────────────────────────────────────
    all_names = {n for s in scenes for n in s["frames"]}
    copied = 0
    copied_reps = 0

    for name in sorted(all_names):
        src = name_to_src.get(name)
        if src is None:
            continue
        if not dry_run:
            dst = out_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
            seen_names.add(name)
        copied += 1

    for scene in scenes:
        rep = scene["representative"]
        src = name_to_src.get(rep)
        if src and not dry_run:
            dst = reps_dir / f"scene{scene['scene_id']:04d}_{rep}"
            if not dst.exists():
                shutil.copy2(src, dst)
        copied_reps += 1

    # ── Сохранение scenes.json ───────────────────────────────────────────────
    if not dry_run:
        scenes_path = out_dir / "scenes.json"
        scenes_path.write_text(
            json.dumps({"scene_gap_sec": scene_gap_sec,
                        "scene_diff": scene_diff,
                        "scenes": scenes},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nscenes.json → {scenes_path}")
        print(f"representatives/ → {reps_dir}  ({copied_reps} кадров)")

    print(f"Кадров в пуле: {copied}")
    return {"scenes": len(scenes), "frames": copied, "representatives": copied_reps}


def _default_sources(classes: list[str] = DEFAULT_CLASSES) -> list[Path]:
    groups = REPO_ROOT / ".data" / "groups"
    sources: list[Path] = []
    # Датасет групп v1
    for cls in classes:
        d = groups / "v1" / "dataset" / cls
        if d.is_dir():
            sources.append(d)
    # Инференс групп: v1 и v2
    for ver in ("v1", "v2"):
        inference_root = groups / ver / "inference" / "images"
        if not inference_root.is_dir():
            continue
        for date_dir in sorted(inference_root.iterdir()):
            for cls in classes:
                d = date_dir / cls
                if d.is_dir():
                    sources.append(d)
    return sources


def main() -> int:
    ap = argparse.ArgumentParser(description="Сбор кропов жителей по сценам")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, metavar="CLS")
    ap.add_argument("--src", type=Path, action="append", dest="sources", metavar="DIR")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".data" / "residents" / "v1" / "scene_pool")
    ap.add_argument("--interval", type=int, default=2, metavar="SEC",
                    help="Временна́я дедупликация (сек). 0 — отключить (default: 2)")
    ap.add_argument("--max-persons", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.35, metavar="F")
    ap.add_argument("--min-blur", type=float, default=40.0, metavar="F")
    ap.add_argument("--scene-gap", type=int, default=120, metavar="SEC",
                    help="Временной разрыв для новой сцены (default: 120s)")
    ap.add_argument("--scene-diff", type=float, default=50.0, metavar="F",
                    help="Pixel-diff для новой сцены (default: 50). 0 — только по времени")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sources = args.sources or _default_sources(args.classes)
    if not sources:
        print("[!] Источники не найдены.", file=sys.stderr)
        return 1

    print(f"Источники ({len(sources)}):")
    for s in sources:
        n = sum(1 for f in s.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS) if s.is_dir() else 0
        print(f"  {s}  [{n}]")
    print(f"Назначение:  {args.out}")
    print(f"Интервал:    {args.interval}s | min_conf: {args.min_conf} | min_blur: {args.min_blur}")
    print(f"Сцена:       gap≥{args.scene_gap}s  ИЛИ  diff≥{args.scene_diff}")
    if args.dry_run:
        print("(dry-run)")
    print()

    collect_scenes(
        sources, args.out,
        collect_classes=set(args.classes),
        max_persons=args.max_persons,
        min_conf=args.min_conf,
        min_blur=args.min_blur,
        interval=args.interval,
        scene_gap_sec=args.scene_gap,
        scene_diff=args.scene_diff,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
