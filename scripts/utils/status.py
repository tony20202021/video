"""
Опрос состояния сервиса и сохранение результатов в .output/status/.

Usage:
    python scripts/utils/status.py
    python scripts/utils/status.py --url http://localhost:8000
    python scripts/utils/status.py --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.time_msk import ts_for_dir, ts_iso

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "status"


def _call_api(url: str) -> dict:
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        return {"error": str(e), "url": url}


def _local_status() -> dict:
    """Статус без запущенного сервера — напрямую из модуля."""
    from backend.routers.status import full_status
    return full_status()


def main() -> None:
    ap = argparse.ArgumentParser(description="Опрос статуса сервиса")
    ap.add_argument("--url", default=None,
                    help="URL бэкенда (напр. http://localhost:8000/status/). "
                         "Если не задан — вызывает локально без HTTP.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--no-save", action="store_true", help="Не сохранять на диск")
    args = ap.parse_args()

    if args.url:
        url = args.url.rstrip("/") + "/status/"
        data = _call_api(url)
    else:
        data = _local_status()

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if not args.no_save:
        run_dir = args.output / f"status_{ts_for_dir()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "report.json"
        report.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт: {report}", file=sys.stderr)


if __name__ == "__main__":
    main()
