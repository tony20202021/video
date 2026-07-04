"""Клиент передачи файлов на Transfer Server.

Все команды отправляют файлы поштучно (не tar.gz) через POST /file.
Адаптивное ограничение CPU: замедляет/ускоряет передачу по соотношению
работа/пауза аналогично YOLO-инференсу.

send  — разово отправить все файлы из каталога run_*.
watch — постоянно следить за каталогом, отправлять появившиеся файлы,
        дожидаться подтверждения и удалять локально.

URL и API-ключ берутся из .env (TRANSFER_SERVER, TRANSFER_API_KEY)
или передаются явно через аргументы.

Usage:
    python scripts/transfer/client.py send .output/pipeline/1_motion_diff/run_XXX
    python scripts/transfer/client.py watch .output/pipeline/1_motion_diff/run_XXX/images
    python scripts/transfer/client.py health
    python scripts/transfer/client.py runs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.adaptive_rate import AdaptiveRateLimiter


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


def _make_limiter(env: dict[str, str]) -> AdaptiveRateLimiter:
    def _ef(k: str, d: float) -> float:
        v = env.get(k, "")
        try:
            return float(v) if v else d
        except ValueError:
            return d

    max_rate = _ef("TRANSFER_MAX_RATE",    5.0)
    min_rate = _ef("TRANSFER_MIN_RATE",    0.1)
    return AdaptiveRateLimiter(
        min_interval=(1.0 / max_rate) if max_rate > 0 else 0.0,
        max_interval=(1.0 / min_rate) if min_rate > 0 else 30.0,
        factor=_ef("TRANSFER_ADAPT_FACTOR", 2.0),
        high=_ef("TRANSFER_ADAPT_HIGH",     0.90),
        low=_ef("TRANSFER_ADAPT_LOW",       0.40),
        window=int(_ef("TRANSFER_ADAPT_WINDOW", 10)),
        label="transfer",
        unit="/с",
    )


def _parse_img_meta(f: Path) -> dict[str, str]:
    """Parse cam, date, time, type from image filename.

    Filename pattern: <cam>_YYYYMMDD_HHMMSS_ffffff_msk_<type>.jpg
    Returns {"cam", "date", "time", "type"} — values may be empty if parsing fails.
    """
    stem = f.stem

    if "_baseline" in stem:
        img_type = "baseline"
    elif "_heartbeat" in stem:
        img_type = "heartbeat"
    elif "_diff" in stem or f.parent.name == "diff":
        img_type = "diff"
    elif "_raw" in stem:
        img_type = "raw"
    else:
        img_type = "unknown"

    cam = date = time_str = ""
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            cam = "_".join(parts[:i])
            date = part
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                time_str = parts[i + 1]
                if i + 2 < len(parts) and len(parts[i + 2]) == 6 and parts[i + 2].isdigit():
                    time_str = f"{time_str}_{parts[i + 2]}"
            break

    return {"cam": cam, "date": date, "time": time_str, "type": img_type}


def _send_file(http, server: str, headers: dict, f: Path, run_root: Path) -> tuple[bool, float]:
    """Отправить один файл. Возвращает (ok, work_ms)."""
    try:
        rel = f.relative_to(run_root).as_posix()
    except ValueError:
        rel = f.name
    t0 = time.monotonic()
    try:
        data = f.read_bytes()
        if not data:
            return True, 0.0
        meta = _parse_img_meta(f)
        headers["X-Rel-Path"] = rel
        headers["X-Cam"]  = meta["cam"]
        headers["X-Date"] = meta["date"]
        headers["X-Time"] = meta["time"]
        headers["X-Type"] = meta["type"]
        resp = http.post(f"{server}/file", content=data, headers=headers)
        resp.raise_for_status()
        ok = resp.json().get("ok", False)
        return ok, (time.monotonic() - t0) * 1000
    except FileNotFoundError:
        return True, (time.monotonic() - t0) * 1000  # удалён между сканом и чтением — нормально
    except Exception as e:
        print(f"  [!] {rel}: {e}", file=sys.stderr)
        return False, (time.monotonic() - t0) * 1000


def cmd_send(args) -> int:
    """Разово отправить все файлы из run_*-каталога на сервер (поштучно)."""
    import httpx

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"[!] Не найдено: {src}", file=sys.stderr)
        return 1

    step   = src.parent.name
    run    = src.name
    server, key = _get_server_and_key(args)
    env    = _load_env()

    exts  = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    files = sorted(f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in exts)

    if not files:
        print(f"  Нет файлов с расширениями {exts} в {src}")
        return 0

    headers = {"X-Api-Key": key, "X-Step": step, "X-Run": run,
               "Content-Type": "application/octet-stream"}

    print(f"  Источник: {src}")
    print(f"  Шаг:     {step}  Прогон: {run}")
    print(f"  Сервер:  {server}")
    print(f"  Файлов:  {len(files)}")
    print()

    limiter = _make_limiter(env)
    n_ok = n_fail = 0

    with httpx.Client(timeout=60.0) as http:
        for i, f in enumerate(files, 1):
            rel = f.relative_to(src).as_posix()
            t0 = time.monotonic()
            ok, work_ms = _send_file(http, server, headers, f, src)
            if ok:
                n_ok += 1
                print(f"  [{i}/{len(files)}] ↑ {rel}  ({work_ms:.0f}мс)")
            else:
                n_fail += 1
            sleep_ms = limiter.sleep(t0)
            msg = limiter.adapt(work_ms, sleep_ms)
            if msg:
                print(msg)

    print(f"\nИтого: {n_ok} отправлено, {n_fail} ошибок")
    return 0 if n_fail == 0 else 1


def cmd_watch(args) -> int:
    """Постоянно следит за каталогом; новые файлы — отправляет, подтверждённые — удаляет."""
    import httpx

    watch_dir = Path(args.watch_dir).resolve()
    run_root  = Path(args.run_root).resolve() if args.run_root else watch_dir.parent
    step      = args.step or run_root.parent.name
    run       = args.run  or run_root.name
    server, key = _get_server_and_key(args)

    env     = _load_env()
    limiter = _make_limiter(env)

    def _ef(k: str, d: float) -> float:
        v = env.get(k, "")
        try:
            return float(v) if v else d
        except ValueError:
            return d

    poll_sec = _ef("TRANSFER_POLL_SEC", 1.0)
    exts     = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    headers  = {"X-Api-Key": key, "X-Step": step, "X-Run": run,
                "Content-Type": "application/octet-stream"}

    print("=== transfer watch ===")
    print(f"  Каталог: {watch_dir}")
    print(f"  Шаг:     {step}  Прогон: {run}")
    print(f"  Сервер:  {server}")
    print(f"  Расш.:   {', '.join(sorted(exts))}")
    _max_r = f"{1/limiter.min_interval:.1f}" if limiter.min_interval > 0 else "∞"
    _min_r = f"{1/limiter.max_interval:.2f}" if limiter.max_interval > 0 else "0"
    print(f"  Скор.:   max={_max_r}/с  min={_min_r}/с  poll={poll_sec}с")
    print()

    n_sent = n_failed = 0
    _fail_count: dict[Path, int] = {}   # consecutive failures per file
    _retry_after: dict[Path, float] = {}  # monotonic time to retry

    with httpx.Client(timeout=60.0) as http:
        while True:
            if not watch_dir.is_dir():
                time.sleep(poll_sec)
                continue

            try:
                new_files = sorted(
                    f for f in watch_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in exts
                )
            except Exception as e:
                print(f"[!] scan: {e}", file=sys.stderr)
                time.sleep(poll_sec)
                continue

            # date dirs present in watch_dir right now (YYYYMMDD subdirs)
            _date_dirs = sorted(
                d.name for d in watch_dir.iterdir()
                if d.is_dir() and d.name.isdigit() and len(d.name) == 8
            ) if watch_dir.is_dir() else []
            _n_dates = len(_date_dirs)

            if new_files:
                print(f"  новый батч: {len(new_files)} файлов")

            now = time.monotonic()
            for _file_idx, f in enumerate(new_files, 1):
                if now < _retry_after.get(f, 0):
                    continue  # backoff not expired yet

                rel = f.relative_to(run_root).as_posix() if f.is_relative_to(run_root) else f.name
                _rel_parts = f.relative_to(watch_dir).parts if f.is_relative_to(watch_dir) else ()
                _date_part = _rel_parts[0] if len(_rel_parts) > 1 and _rel_parts[0] in _date_dirs else ""

                t0 = time.monotonic()
                try:
                    data = f.read_bytes()
                    if not data:
                        continue
                    _meta = _parse_img_meta(f)
                    headers["X-Rel-Path"] = rel
                    headers["X-Cam"]  = _meta["cam"]
                    headers["X-Date"] = _meta["date"]
                    headers["X-Time"] = _meta["time"]
                    headers["X-Type"] = _meta["type"]
                    resp = http.post(f"{server}/file", content=data, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()
                    work_ms = (time.monotonic() - t0) * 1000
                    if result.get("ok"):
                        f.unlink(missing_ok=True)
                        _fail_count.pop(f, None)
                        _retry_after.pop(f, None)
                        n_sent += 1
                        _date_tag = f"  [дата {_date_part} / всего: {_n_dates}]" if _date_part else ""
                        print(f"  [в батче {_file_idx}/{len(new_files)}] ↑ {rel}"
                              f"  ({len(data)/1024:.1f} КБ  {work_ms:.0f}мс)"
                              f"  всего={n_sent}{_date_tag}")
                    else:
                        n_failed += 1
                        work_ms = (time.monotonic() - t0) * 1000
                        print(f"  [!] нет подтверждения: {rel}", file=sys.stderr)
                except FileNotFoundError:
                    work_ms = (time.monotonic() - t0) * 1000
                    _fail_count.pop(f, None)
                    _retry_after.pop(f, None)
                except Exception as e:
                    n_failed += 1
                    work_ms = (time.monotonic() - t0) * 1000
                    cnt = _fail_count.get(f, 0) + 1
                    _fail_count[f] = cnt
                    delay = min(2.0 ** cnt, 60.0)  # 2s, 4s, 8s, …, 60s
                    _retry_after[f] = time.monotonic() + delay
                    print(f"  [!] {rel}: {e}  (retry in {delay:.0f}s)", file=sys.stderr)

                sleep_ms = limiter.sleep(t0)
                msg = limiter.adapt(work_ms, sleep_ms)
                if msg:
                    print(msg)

            if not new_files:
                time.sleep(poll_sec)


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
    ap = argparse.ArgumentParser(description="Transfer client (per-file)")
    ap.add_argument("--server", default=None, help="URL сервера (или TRANSFER_SERVER в .env)")
    ap.add_argument("--key",    default=None, help="API-ключ (или TRANSFER_API_KEY в .env)")

    sub = ap.add_subparsers(dest="cmd")

    p_send = sub.add_parser("send", help="Отправить все файлы из run_*-каталога (разово)")
    p_send.add_argument("src", help="Путь к run_*-каталогу")
    p_send.add_argument("--ext", default="jpg", metavar="EXTS",
                        help="Расширения файлов через запятую (default: jpg)")

    p_watch = sub.add_parser("watch", help="Следить за каталогом и отправлять новые файлы")
    p_watch.add_argument("watch_dir", help="Каталог для наблюдения (напр. run_XXX/images)")
    p_watch.add_argument("--run-root", default="",
                         help="Корень прогона (default: parent watch_dir = run_XXX)")
    p_watch.add_argument("--step",     default="",
                         help="Имя шага (default: parent run_root)")
    p_watch.add_argument("--run",      default="",
                         help="Имя прогона (default: run_root.name)")
    p_watch.add_argument("--ext",      default="jpg", metavar="EXTS",
                         help="Расширения файлов через запятую (default: jpg)")

    sub.add_parser("health", help="Проверить доступность сервера")
    sub.add_parser("runs",   help="Список принятых прогонов на сервере")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 1

    if args.cmd == "send":
        return cmd_send(args)
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "health":
        return cmd_health(args)
    if args.cmd == "runs":
        return cmd_runs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
