"""Версия проекта — хранится в файле VERSION (корень репо), формат MAJOR.MINOR.PATCH.

  python scripts/version.py                 # показать текущую
  python scripts/version.py bump            # минор +1, patch=0   (0.1.2 → 0.2.0)
  python scripts/version.py bump --patch    # patch +1            (0.1.2 → 0.1.3)
  python scripts/version.py bump --major    # мажор +1, минор=patch=0 (0.1.2 → 1.0.0)
  python scripts/version.py set 1.2.0       # задать явно

Правило проекта: на каждое изменение — минор +1 (`bump`); мажор — только когда явно сказано.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.is_file() else "0.0.0"


def write_version(v: str) -> None:
    VERSION_FILE.write_text(v + "\n", encoding="utf-8")


def parse(v: str) -> tuple[int, int, int]:
    parts = (v.strip().split(".") + ["0", "0", "0"])[:3]
    return int(parts[0]), int(parts[1]), int(parts[2])


def next_version(current: str, kind: str) -> str:
    major, minor, patch = parse(current)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return f"{major}.{minor + 1}.0"  # minor (по умолчанию)


def main() -> int:
    ap = argparse.ArgumentParser(description="Версия проекта (файл VERSION)")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bump", help="увеличить версию")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--major", action="store_const", const="major", dest="kind")
    g.add_argument("--patch", action="store_const", const="patch", dest="kind")
    st = sub.add_parser("set", help="задать версию явно")
    st.add_argument("value")
    args = ap.parse_args()

    if args.cmd == "bump":
        old = read_version()
        new = next_version(old, args.kind or "minor")
        write_version(new)
        print(f"{old} → {new}")
    elif args.cmd == "set":
        write_version(args.value.strip())
        print(read_version())
    else:
        print(read_version())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
