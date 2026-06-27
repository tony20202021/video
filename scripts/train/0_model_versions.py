"""CLI для управления версиями моделей. Логика — в src/ml/versions.py."""

from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ml.versions import (
    list_versions, get_active_version, activate_version,
    register_version, clean_versions,
)


def cmd_list(_) -> int:
    versions = list_versions()
    active = get_active_version()
    if not versions:
        print("Нет версий в models/classify/")
        return 0
    print(f"{'Ver':>4}  {'Size':>6}  {'F1':>6}  {'Acc':>6}  {'Создана':>20}  Заметки")
    print("─" * 72)
    for v in versions:
        m = v.get("metrics", {})
        f1  = f"{m.get('f1', 0):.3f}" if m.get("f1") else "—"
        acc = f"{m.get('accuracy', 0):.3f}" if m.get("accuracy") else "—"
        flag = " ← ACTIVE" if v["version"] == active else ""
        print(f"  v{v['version']:2d}  {v['size_kb']:5d}К  {f1:>6}  {acc:>6}"
              f"  {v.get('created_at','')[:19]:>20}  {v.get('notes','')[:25]}{flag}")
    return 0


def cmd_info(args) -> int:
    v = int(args.version)
    from ml.versions import model_path, manifest_path
    mp = model_path(v)
    if not mp.is_file():
        print(f"Нет модели v{v}", file=sys.stderr); return 1
    print(f"Файл: {mp}  ({mp.stat().st_size//1024} KB)")
    mf = manifest_path(v)
    if mf.is_file():
        import json
        print(json.dumps(json.loads(mf.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


def cmd_activate(args) -> int:
    v = int(args.version)
    ok = activate_version(v)
    if ok: print(f"Активирована v{v}")
    else: print(f"Ошибка активации v{v}", file=sys.stderr)
    return 0 if ok else 1


def cmd_register(args) -> int:
    src = Path(args.source)
    if not src.is_file():
        print(f"Нет файла: {src}", file=sys.stderr); return 1
    n = register_version(src, notes=args.notes or "")
    print(f"Зарегистрирована версия v{n}")
    return 0


def cmd_clean(args) -> int:
    removed = clean_versions(args.keep)
    print(f"Удалены: {removed}" if removed else "Нечего удалять")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Управление версиями ML-моделей")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list")
    si = sub.add_parser("info");     si.add_argument("version")
    sa = sub.add_parser("activate"); sa.add_argument("version")
    sr = sub.add_parser("register"); sr.add_argument("source"); sr.add_argument("--notes", default="")
    sc = sub.add_parser("clean");    sc.add_argument("--keep", type=int, default=3)
    args = p.parse_args()
    if not args.command: p.print_help(); return 0
    return {"list": cmd_list, "info": cmd_info, "activate": cmd_activate,
            "register": cmd_register, "clean": cmd_clean}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
