"""Клиент передачи run_*-каталогов на сервер.

Упаковывает run_*-каталог в tar.gz и отправляет на Transfer Server.
Шаг пайплайна (step) определяется автоматически по имени родительского каталога.

URL и API-ключ берутся из .env (TRANSFER_SERVER, TRANSFER_API_KEY)
или передаются явно через аргументы.

Usage:
    python scripts/transfer/client.py send .output/pipeline/1_motion_diff/run_20260629_XXX
    python scripts/transfer/client.py send run_XXX --server http://1.2.3.4:8765 --key SECRET
    python scripts/transfer/client.py runs                   # список принятых run на сервере
    python scripts/transfer/client.py health                 # проверка сервера
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> dict[str, str]:
    env_file = REPO_ROOT / ".env"
    result: dict[str, str] = {}
    if not env_file.is_file():
        return result
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _get_server_and_key(args) -> tuple[str, str]:
    env = _load_env()
    server = args.server or os.environ.get("TRANSFER_SERVER") or env.get("TRANSFER_SERVER", "")
    key    = args.key    or os.environ.get("TRANSFER_API_KEY") or env.get("TRANSFER_API_KEY", "")
    if not server:
        print("[!] Укажи --server или задай TRANSFER_SERVER в .env", file=sys.stderr)
        sys.exit(1)
    return server.rstrip("/"), key


def _pack(src: Path) -> Path:
    """Упаковать src в tar.gz во временный файл. Возвращает путь к файлу."""
    tmp = Path(tempfile.mktemp(suffix=".tar.gz"))
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    return tmp


def _upload(server: str, step: str, tmp: Path, key: str) -> dict:
    import httpx

    size = tmp.stat().st_size
    print(f"  Размер архива: {size / 1_048_576:.1f} МБ")

    def _gen():
        done = 0
        with open(tmp, "rb") as f:
            while chunk := f.read(1 << 20):  # 1 МБ
                done += len(chunk)
                pct = done * 100 // size
                print(f"\r  Отправка: {pct:3d}%  ({done / 1_048_576:.1f} / {size / 1_048_576:.1f} МБ)",
                      end="", flush=True)
                yield chunk
        print()

    headers = {"X-Api-Key": key, "Content-Type": "application/octet-stream"}
    with httpx.Client(timeout=None) as client:
        resp = client.post(f"{server}/pipeline/{step}", content=_gen(), headers=headers)
    resp.raise_for_status()
    return resp.json()


def cmd_send(args) -> int:
    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"[!] Не найдено: {src}", file=sys.stderr)
        return 1

    step = src.parent.name  # 1_motion_diff / 2_yolo_boxes_files / ...
    server, key = _get_server_and_key(args)

    print(f"  Источник: {src}")
    print(f"  Шаг:      {step}")
    print(f"  Сервер:   {server}")

    tmp = _pack(src)
    try:
        result = _upload(server, step, tmp, key)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"  OK → {result.get('dest', '?')}")
    return 0


def cmd_health(args) -> int:
    import httpx
    server, _ = _get_server_and_key(args)
    try:
        r = httpx.get(f"{server}/health", timeout=5)
        r.raise_for_status()
        print(f"  {server}  →  {r.json()}")
        return 0
    except Exception as e:
        print(f"[!] {e}", file=sys.stderr)
        return 1


def cmd_runs(args) -> int:
    import httpx
    server, key = _get_server_and_key(args)
    r = httpx.get(f"{server}/runs", headers={"X-Api-Key": key}, timeout=10)
    r.raise_for_status()
    data: dict = r.json()
    if not data:
        print("  (нет принятых прогонов)")
        return 0
    for step, runs in sorted(data.items()):
        print(f"  {step}:")
        for run in runs:
            print(f"    {run}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Transfer client")
    ap.add_argument("--server", default=None, help="URL сервера (или TRANSFER_SERVER в .env)")
    ap.add_argument("--key",    default=None, help="API-ключ (или TRANSFER_API_KEY в .env)")

    sub = ap.add_subparsers(dest="cmd")

    p_send = sub.add_parser("send", help="Отправить run_*-каталог на сервер")
    p_send.add_argument("src", help="Путь к run_*-каталогу")

    sub.add_parser("health", help="Проверить доступность сервера")
    sub.add_parser("runs",   help="Список принятых прогонов на сервере")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 1

    if args.cmd == "send":
        return cmd_send(args)
    if args.cmd == "health":
        return cmd_health(args)
    if args.cmd == "runs":
        return cmd_runs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
