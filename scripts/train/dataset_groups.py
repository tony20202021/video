"""
Управление датасетом для Модели 1 (классификатор групп).

Структура датасета:
  .data/groups/v<N>/
    dataset.json        — метаданные: классы, счётчики, источник
    1_resident/         ← файлы по классу
    2_delivery/
    3_utilities/
    skip/               ← намеренно пропущены при разметке
    unknown/            ← неизвестный класс
    new/                ← новые кропы без разметки (→ 2_label_ui → apply)

Команды:
  build   --labels PATH [--out PATH] [--version vN]
          Читает labels.json, обновляет устаревшие пути, копирует файлы по классам.
          Используется для первичной сборки датасета из .output/train/2_label_ui/labels.json.

  apply   --labels PATH --dataset PATH [--move]
          Читает labels.json (обычно из new/), копирует/перемещает файлы
          в подпапки датасета по меткам. Обновляет dataset.json.

  add     --src PATH --dataset PATH
          Копирует кропы из прогона pipeline/2_yolo_boxes_files в new/.
          Файлы-дубли (по имени) пропускает.

  check   --src PATH --dataset PATH
          Проверяет файлы из src на дубли с любым файлом в датасете (по имени).
          Выводит новые, дубли, итог. Ничего не копирует.

  status  --dataset PATH
          Показывает счётчики по классам и дубли имён файлов.

Usage:
    python scripts/train/dataset_groups.py build --labels .output/train/2_label_ui/labels.json
    python scripts/train/dataset_groups.py apply --labels .data/groups/v1/new/labels.json --dataset .data/groups/v1
    python scripts/train/dataset_groups.py apply --labels .data/groups/v1/new/labels.json --dataset .data/groups/v1 --move
    python scripts/train/dataset_groups.py check --src .data/groups/new --dataset .data/groups/v1
    python scripts/train/dataset_groups.py add --src .output/pipeline/2_yolo_boxes_files/run_xxx --dataset .data/groups/v1
    python scripts/train/dataset_groups.py status --dataset .data/groups/v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.classes import GROUP_CLASSES as MAIN_CLASSES

DEFAULT_DATA = REPO_ROOT / ".data" / "groups"
DEFAULT_LABELS = REPO_ROOT / ".output" / "train" / "2_label_ui" / "labels.json"
MSK = timezone(timedelta(hours=3))
EXTRA_CLASSES = ["skip", "unknown", "new"]
ALL_CLASSES = MAIN_CLASSES + EXTRA_CLASSES

# Устаревшие префиксы путей → новые (для миграции)
PATH_MIGRATIONS = [
    (".output\\cameras\\5_2_yolo_boxes_files", ".output\\pipeline\\2_yolo_boxes_files"),
    (".output/cameras/5_2_yolo_boxes_files",   ".output/pipeline/2_yolo_boxes_files"),
    (".output\\cameras\\6_2_classify_groups_files", ".output\\pipeline\\3_classify_groups"),
    (".output/cameras/6_2_classify_groups_files",   ".output/pipeline/3_classify_groups"),
]

# Устаревшие имена классов → новые
CLASS_MIGRATIONS = {
    "resident":  "1_resident",
    "delivery":  "2_delivery",
    "utilities": "3_utilities",
    "other":     None,   # удалён — перенести в unknown
    "courier":   None,   # удалён — перенести в unknown
}


def _fix_path(p: str) -> str:
    for old, new in PATH_MIGRATIONS:
        if old in p:
            p = p.replace(old, new)
    return p


def _fix_class(cls: str) -> str:
    """Переводит устаревшее имя класса в новое. courier → unknown."""
    if cls in CLASS_MIGRATIONS:
        return CLASS_MIGRATIONS[cls] or "unknown"
    return cls


def _collect_existing_names(dataset_dir: Path) -> set[str]:
    """Собирает все имена файлов из всех подпапок датасета."""
    names: set[str] = set()
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file():
                    names.add(f.name)
    return names


def _prune_empty(dirs: set[Path]) -> None:
    """Удаляет каталоги из набора, если они пустые после переноса файлов."""
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def cmd_build(args) -> int:
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = REPO_ROOT / labels_path
    if not labels_path.exists():
        print(f"[!] labels.json не найден: {labels_path}", file=sys.stderr)
        return 1

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = raw.get("labels", {})

    # Определяем выходной каталог
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        version = args.version
        if not version:
            existing = sorted(DEFAULT_DATA.glob("v*/dataset.json"))
            version = f"v{len(existing) + 1}"
        out_dir = DEFAULT_DATA / version

    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in ALL_CLASSES:
        (out_dir / cls).mkdir(exist_ok=True)

    counts: dict[str, int] = {cls: 0 for cls in ALL_CLASSES}
    missing = 0
    migrated = 0

    for rel_path, cls in labels.items():
        fixed = _fix_path(rel_path)
        if fixed != rel_path:
            migrated += 1

        src = REPO_ROOT / fixed
        if not src.exists():
            missing += 1
            continue

        cls_fixed = _fix_class(cls)
        cls_dir = cls_fixed if cls_fixed in ALL_CLASSES else "unknown"
        dst = out_dir / cls_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[cls_dir] = counts.get(cls_dir, 0) + 1

    dataset_meta = {
        "version": 1,
        "classes": MAIN_CLASSES,
        "extra_classes": EXTRA_CLASSES,
        "created_at": datetime.now(MSK).isoformat(timespec="seconds"),
        "source_labels": str(labels_path.relative_to(REPO_ROOT)),
        "counts": {k: v for k, v in counts.items() if v > 0},
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Датасет: {out_dir}")
    for cls, cnt in counts.items():
        if cnt:
            print(f"  {cls}: {cnt}")
    if missing:
        print(f"  [!] Не найдено файлов: {missing}")
    if migrated:
        print(f"  Исправлено путей: {migrated}")
    return 0


def cmd_apply(args) -> int:
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = REPO_ROOT / labels_path
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    if not labels_path.exists():
        print(f"[!] labels.json не найден: {labels_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[!] Датасет не найден: {dataset_dir}", file=sys.stderr)
        return 1

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = raw.get("labels", {})

    for cls in ALL_CLASSES:
        (dataset_dir / cls).mkdir(exist_ok=True)

    counts: dict[str, int] = {}
    missing = 0
    moved_dirs: set[Path] = set()

    for rel_path, cls in labels.items():
        src = REPO_ROOT / rel_path
        if not src.exists():
            # файл мог быть указан с другим базовым путём — ищем по имени в new/
            fallback = dataset_dir / "new" / Path(rel_path).name
            if fallback.exists():
                src = fallback
            else:
                print(f"  [!] Не найден: {rel_path}")
                missing += 1
                continue

        cls_fixed = _fix_class(cls)
        cls_dir = cls_fixed if cls_fixed in ALL_CLASSES else "unknown"
        dst = dataset_dir / cls_dir / src.name
        if not dst.exists():
            if args.move:
                shutil.move(str(src), dst)
                moved_dirs.add(src.parent)
            else:
                shutil.copy2(src, dst)
        counts[cls_dir] = counts.get(cls_dir, 0) + 1

    if moved_dirs:
        _prune_empty(moved_dirs)

    # Обновляем dataset.json
    meta_path = dataset_dir / "dataset.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing_counts: dict = meta.get("counts", {})
        for k, v in counts.items():
            existing_counts[k] = existing_counts.get(k, 0) + v
        meta["counts"] = {k: v for k, v in existing_counts.items() if v > 0}
        meta["updated_at"] = datetime.now(MSK).isoformat(timespec="seconds")
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    action = "Перемещено" if args.move else "Скопировано"
    print(f"Датасет: {dataset_dir}")
    for cls_dir, cnt in sorted(counts.items()):
        print(f"  {cls_dir}/: {cnt}")
    if missing:
        print(f"  [!] Не найдено файлов: {missing}")
    print(f"{action}: {sum(counts.values())}")
    return 0


def cmd_check(args) -> int:
    src_dir = Path(args.src)
    if not src_dir.is_absolute():
        src_dir = REPO_ROOT / src_dir
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    if not src_dir.exists():
        print(f"[!] Не найдено: {src_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[!] Датасет не найден: {dataset_dir}", file=sys.stderr)
        return 1

    unique_dir = src_dir / "unique"
    double_dir = src_dir / "double"

    # Собираем имена файлов из датасета: name → [class, ...]
    dataset_names: dict[str, list[str]] = {}
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.is_file():
                dataset_names.setdefault(f.name, []).append(subdir.name)

    IMAGE_EXTS = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    # Ищем только в корне и подпапках, исключая unique/ и double/
    src_files: list[Path] = []
    for f in sorted(src_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        # пропускаем файлы внутри unique/ и double/
        try:
            rel = f.relative_to(src_dir)
            if rel.parts[0] in ("unique", "double"):
                continue
        except ValueError:
            pass
        src_files.append(f)

    n_unique = 0
    n_double = 0
    moved_dirs: set[Path] = set()

    for f in src_files:
        rel = f.relative_to(src_dir)
        where = dataset_names.get(f.name)
        if where:
            dst = double_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dst)
            n_double += 1
        else:
            dst = unique_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dst)
            n_unique += 1
        if f.parent != src_dir:
            moved_dirs.add(f.parent)

    _prune_empty(moved_dirs)

    print(f"Источник:  {src_dir}  ({len(src_files)} файлов)")
    print(f"Датасет:   {dataset_dir}  ({len(dataset_names)} файлов во всех классах)")
    print()
    print(f"unique/:   {n_unique}")
    print(f"double/:   {n_double}")
    print(f"Итого:     {len(src_files)}")

    return 0


def cmd_add(args) -> int:
    src_dir = Path(args.src)
    if not src_dir.is_absolute():
        src_dir = REPO_ROOT / src_dir
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    if not src_dir.exists():
        print(f"[!] Не найдено: {src_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[!] Датасет не найден: {dataset_dir}", file=sys.stderr)
        return 1

    new_dir = dataset_dir / "new"
    new_dir.mkdir(exist_ok=True)

    existing = _collect_existing_names(dataset_dir)

    exts = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    crops = sorted(f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    copied = 0
    skipped_dupe = 0

    for crop in crops:
        if crop.name in existing:
            skipped_dupe += 1
            continue
        shutil.copy2(crop, new_dir / crop.name)
        existing.add(crop.name)
        copied += 1

    print(f"Добавлено в new/: {copied}")
    print(f"Пропущено дублей: {skipped_dupe}")
    return 0


def cmd_status(args) -> int:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    if not dataset_dir.exists():
        print(f"[!] Не найдено: {dataset_dir}", file=sys.stderr)
        return 1

    meta_path = dataset_dir / "dataset.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"Датасет: {dataset_dir.name}  (создан {meta.get('created_at', '?')})")
    else:
        print(f"Датасет: {dataset_dir.name}  (нет dataset.json)")

    all_names: dict[str, list[str]] = {}  # name → [class, ...]
    total = 0

    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        files = [f for f in subdir.iterdir() if f.is_file()]
        total += len(files)
        print(f"  {subdir.name}/: {len(files)}")
        for f in files:
            all_names.setdefault(f.name, []).append(subdir.name)

    print(f"Итого: {total}")

    dupes = {name: classes for name, classes in all_names.items() if len(classes) > 1}
    if dupes:
        print(f"\n[!] Дубли ({len(dupes)}):")
        for name, classes in sorted(dupes.items()):
            print(f"  {name} → {', '.join(classes)}")
    else:
        print("Дублей нет.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Управление датасетом групп (Модель 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Собрать датасет из labels.json")
    p_build.add_argument("--labels", default=str(DEFAULT_LABELS))
    p_build.add_argument("--out", default="")
    p_build.add_argument("--version", default="")

    p_apply = sub.add_parser("apply", help="Разложить файлы из new/ по классам по labels.json")
    p_apply.add_argument("--labels",  required=True, help="Файл разметки (labels.json)")
    p_apply.add_argument("--dataset", required=True, help="Каталог датасета (.data/groups/v1)")
    p_apply.add_argument("--move", action="store_true", help="Переместить (не копировать)")

    p_check = sub.add_parser("check", help="Проверить файлы на дубли с датасетом (без копирования)")
    p_check.add_argument("--src",     required=True, help="Каталог новых файлов")
    p_check.add_argument("--dataset", required=True, help="Каталог датасета (.data/groups/v1)")
    p_check.add_argument("--ext", default="jpg",
                         help="Расширения файлов через запятую (default: jpg)")

    p_add = sub.add_parser("add", help="Добавить новые кропы в new/")
    p_add.add_argument("--src", required=True, help="Каталог прогона 2_yolo_boxes_files")
    p_add.add_argument("--dataset", required=True, help="Каталог датасета (.data/groups/v1)")
    p_add.add_argument("--ext", default="jpg",
                       help="Расширения файлов через запятую (default: jpg)")

    p_status = sub.add_parser("status", help="Показать состояние датасета")
    p_status.add_argument("--dataset", required=True)

    args = parser.parse_args()
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "apply":
        return cmd_apply(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "status":
        return cmd_status(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
