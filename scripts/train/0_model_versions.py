"""CLI для управления версиями моделей. Логика — в src/ml/versions.py."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ml.versions import (
    activate_tag,
    classify_manifest_path,
    classify_model_path,
    clean_versions,
    get_active_tag,
    list_versions,
    register_version,
)


def cmd_list(_) -> int:
    versions = list_versions()
    active = get_active_tag()
    if not versions:
        print("Нет версий в .models/classify/")
        return 0
    print(f"{'Tag':<8} {'DS':<6} {'Size':>6}  {'Acc':>6}  {'Создана':>20}  Заметки")
    print("─" * 78)
    for v in versions:
        m = v.get("metrics", {})
        acc = m.get("best_val_acc") or m.get("accuracy")
        acc_s = f"{acc:.3f}" if acc else "—"
        ds = v.get("dataset_version") or "—"
        flag = " ← ACTIVE" if v["tag"] == active else ""
        print(
            f"  {v['tag']:<6} {ds:<6} {v['size_kb']:5d}К  {acc_s:>6}"
            f"  {v.get('created_at', '')[:19]:>20}  {v.get('notes', '')[:25]}{flag}"
        )
    return 0


def cmd_info(args) -> int:
    tag = args.version
    mp = classify_model_path(tag)
    if not mp.is_file():
        print(f"Нет модели {tag}", file=sys.stderr)
        return 1
    print(f"Файл: {mp}  ({mp.stat().st_size // 1024} KB)")
    mf = classify_manifest_path(tag)
    if mf.is_file():
        print(json.dumps(json.loads(mf.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


def cmd_activate(args) -> int:
    tag = args.version
    ok = activate_tag(tag)
    if ok:
        print(f"Активирована {tag}")
    else:
        print(f"Ошибка активации {tag}", file=sys.stderr)
    return 0 if ok else 1


def cmd_register(args) -> int:
    src = Path(args.source)
    if not src.is_file():
        print(f"Нет файла: {src}", file=sys.stderr)
        return 1
    tag = register_version(src, notes=args.notes or "")
    print(f"Зарегистрирована версия {tag}")
    return 0


def cmd_clean(args) -> int:
    removed = clean_versions(args.keep)
    print(f"Удалены: {removed}" if removed else "Нечего удалять")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Управление версиями ML-моделей")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list")
    si = sub.add_parser("info")
    si.add_argument("version", help="Тег модели, напр. v1_2 или v3")
    sa = sub.add_parser("activate")
    sa.add_argument("version", help="Тег модели, напр. v1_2 или v3")
    sr = sub.add_parser("register")
    sr.add_argument("source")
    sr.add_argument("--notes", default="")
    sc = sub.add_parser("clean")
    sc.add_argument("--keep", type=int, default=3)
    args = p.parse_args()
    if not args.command:
        p.print_help()
        return 0
    return {
        "list": cmd_list,
        "info": cmd_info,
        "activate": cmd_activate,
        "register": cmd_register,
        "clean": cmd_clean,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
