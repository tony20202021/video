"""Авторазметка кадров внутри сцены по одному размеченному кадру.

Workflow:
  1. В collect_scenes.py собран scene_pool/ + scenes.json + representatives/
  2. Пользователь разметил representatives/ в labels.json (1 кадр на сцену)
  3. Этот скрипт читает labels.json (частичный) + scenes.json
     → автоматически присваивает тот же person_id всем кадрам той же сцены
  4. Опционально: проверяет pixel-diff внутри сцены (--intra-diff-limit)
     для отлова кадров где в рамках «сцены» реально сменился человек

Usage:
    python scripts/train/propagate_scenes.py .data/residents/v0/scene_pool
    python scripts/train/propagate_scenes.py .data/residents/v0/scene_pool --intra-diff 30
    python scripts/train/propagate_scenes.py .data/residents/v0/scene_pool --dry-run
    python scripts/train/propagate_scenes.py .data/residents/v0/scene_pool --report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _diff(a_path: Path, b_path: Path) -> float:
    a = cv2.imread(str(a_path))
    b = cv2.imread(str(b_path))
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


_RE_SCENE_PREFIX = re.compile(r"^scene\d+_")


def _strip_prefix(name: str) -> str:
    """Убирает префикс scene0042_ который добавляется при копировании в representatives/."""
    return _RE_SCENE_PREFIX.sub("", name)


def _load_labels(pool_dir: Path) -> dict[str, str]:
    """Читает labels.json из pool_dir. Формат: {abs_path|name → person_id}.

    Поддерживает два варианта имён:
      - оригинальные (из scene_pool/*.jpg)
      - с префиксом scene0042_ (из scene_pool/representatives/*.jpg)
    Префикс снимается автоматически чтобы оба варианта давали один и тот же ключ.
    """
    lf = pool_dir / "labels.json"
    if not lf.exists():
        return {}
    raw = json.loads(lf.read_text(encoding="utf-8"))
    data = raw.get("labels", raw)
    return {_strip_prefix(Path(k).name): v for k, v in data.items() if isinstance(v, str)}


def _save_labels(pool_dir: Path, labels: dict[str, str]) -> None:
    lf = pool_dir / "labels.json"
    existing_raw: dict = {}
    if lf.exists():
        existing_raw = json.loads(lf.read_text(encoding="utf-8"))

    if "labels" in existing_raw:
        existing_raw["labels"].update(labels)
        out = existing_raw
    else:
        existing_raw.update(labels)
        out = existing_raw

    lf.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def propagate(
    pool_dir: Path,
    *,
    intra_diff_limit: float = 0.0,
    dry_run: bool = False,
    report: bool = False,
) -> dict:
    scenes_path = pool_dir / "scenes.json"
    if not scenes_path.exists():
        print(f"[!] scenes.json не найден: {scenes_path}", file=sys.stderr)
        return {}

    scenes_data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes: list[dict] = scenes_data["scenes"]

    labels = _load_labels(pool_dir)
    print(f"Загружено меток: {len(labels)}")

    # Инвертированный индекс: имя файла → scene_id
    file_to_scene: dict[str, int] = {}
    for s in scenes:
        for name in s["frames"]:
            file_to_scene[name] = s["scene_id"]

    # Сгруппировать метки по scene_id
    scene_labels: dict[int, dict[str, str]] = {}  # scene_id → {name: pid}
    for name, pid in labels.items():
        if pid in ("skip", "excluded"):
            continue
        sid = file_to_scene.get(name)
        if sid is None:
            continue
        scene_labels.setdefault(sid, {})[name] = pid

    # Для каждой сцены — найти pid (из помеченных кадров)
    n_propagated = 0
    n_conflicts = 0
    n_diff_rejected = 0
    n_already = 0
    n_unlabeled_scenes = 0
    new_labels: dict[str, str] = {}
    conflict_report: list[str] = []

    for scene in scenes:
        sid = scene["scene_id"]
        manual = scene_labels.get(sid, {})

        if not manual:
            n_unlabeled_scenes += 1
            continue

        # Проверяем конфликт: разные person_id в одной сцене
        pids = set(manual.values())
        if len(pids) > 1:
            n_conflicts += 1
            conflict_report.append(
                f"  сцена {sid} ({scene['cam']} {scene['date']} "
                f"start={scene['start_time']}): {pids}"
            )
            # При конфликте не распространяем — оставляем явно размеченным
            continue

        pid = next(iter(pids))
        labeled_name = next(iter(manual.keys()))
        labeled_path = pool_dir / labeled_name

        for name in scene["frames"]:
            if name in labels:
                n_already += 1
                continue

            if intra_diff_limit > 0:
                p = pool_dir / name
                if p.exists() and labeled_path.exists():
                    d = _diff(labeled_path, p)
                    if d > intra_diff_limit:
                        n_diff_rejected += 1
                        if report:
                            print(f"  REJECTED diff={d:.1f}: {name}")
                        continue

            new_labels[name] = pid
            n_propagated += 1

    print(f"\nСцен всего:         {len(scenes)}")
    print(f"  Размечено (вручную): {len(scenes) - n_unlabeled_scenes - n_conflicts}")
    print(f"  Конфликты (пропущены): {n_conflicts}")
    print(f"  Без метки:           {n_unlabeled_scenes}")
    print(f"\nКадров авторазмечено:  {n_propagated}")
    print(f"  Уже было:            {n_already}")
    print(f"  Отброшено (diff):     {n_diff_rejected}")

    if conflict_report:
        print("\nСцены с конфликтами (несколько person_id в одной сцене):")
        for line in conflict_report:
            print(line)
        print("  → Разметьте вручную или разбейте сцену (уменьшите --scene-diff в collect_scenes.py)")

    if new_labels and not dry_run:
        _save_labels(pool_dir, new_labels)
        print(f"\nОбновлён labels.json: +{len(new_labels)} записей")
    elif dry_run:
        print(f"\n(dry-run) Добавилось бы: {len(new_labels)}")

    return {
        "propagated": n_propagated,
        "conflicts": n_conflicts,
        "diff_rejected": n_diff_rejected,
        "unlabeled_scenes": n_unlabeled_scenes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Авторазметка кадров внутри сцен")
    ap.add_argument("pool_dir", type=Path, help="Каталог scene_pool (с scenes.json)")
    ap.add_argument("--intra-diff", type=float, default=0.0, metavar="F",
                    help="Лимит pixel-diff внутри сцены. 0 — не проверять. (default: 0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Не писать labels.json — только показать результат")
    ap.add_argument("--report", action="store_true",
                    help="Подробный вывод отброшенных по diff")
    args = ap.parse_args()

    if not args.pool_dir.is_dir():
        print(f"[!] Каталог не найден: {args.pool_dir}", file=sys.stderr)
        return 1

    print(f"Пул: {args.pool_dir}")
    if args.intra_diff > 0:
        print(f"Лимит intra-diff: {args.intra_diff}")
    if args.dry_run:
        print("(dry-run)")
    print()

    propagate(args.pool_dir, intra_diff_limit=args.intra_diff,
              dry_run=args.dry_run, report=args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
