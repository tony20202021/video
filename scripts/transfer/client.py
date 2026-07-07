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
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.access import load_env_file
from common.utils.adaptive_rate import AdaptiveRateLimiter
from common.utils.log_setup import setup_logging

logger = logging.getLogger(__name__)


def _load_env() -> dict[str, str]:
    return load_env_file(REPO_ROOT / ".env")


def _get_server_and_key(args) -> tuple[str, str]:
    env = _load_env()
    server = args.server or os.environ.get("TRANSFER_SERVER") or env.get("TRANSFER_SERVER", "")
    key    = args.key    or os.environ.get("TRANSFER_API_KEY") or env.get("TRANSFER_API_KEY", "")
    if not server:
        logger.error("Укажи --server или задай TRANSFER_SERVER в .env")
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


def _find_run_name(parts: tuple[str, ...]) -> str:
    for p in parts:
        if p.startswith("run_"):
            return p
    return ""


def _resolve_routing(
    file_path: Path,
    *,
    watch_dir: Path | None = None,
    run_root: Path | None = None,
    parent: str = "",
    step: str = "",
    run: str = "",
) -> dict[str, str]:
    """parent/step/run/rel_path для одного файла.

    WatchDir = .../run_XXX/images  → run фиксирован из каталога.
    WatchDir = .../1_motion_diff/  → run из сегмента run_* в пути файла.
    """
    file_path = file_path.resolve()

    if watch_dir is not None:
        watch_dir = watch_dir.resolve()
        if watch_dir.name == "images":
            run_dir = watch_dir.parent
            step_dir = run_dir.parent
            resolved_parent = parent or step_dir.parent.name
            resolved_step = step or step_dir.name
            resolved_run = run or run_dir.name
            rel_base = watch_dir
        else:
            resolved_parent = parent or watch_dir.parent.name
            resolved_step = step or watch_dir.name
            rel_parts = (
                file_path.relative_to(watch_dir).parts
                if file_path.is_relative_to(watch_dir)
                else ()
            )
            resolved_run = run or _find_run_name(rel_parts)
            rel_base = watch_dir
    elif run_root is not None:
        run_root = run_root.resolve()
        resolved_parent = parent or run_root.parent.parent.name
        resolved_step = step or run_root.parent.name
        resolved_run = run or run_root.name
        rel_base = run_root
    else:
        resolved_parent = parent
        resolved_step = step
        resolved_run = run
        rel_base = file_path.parent

    try:
        rel_path = file_path.relative_to(rel_base).as_posix()
    except ValueError:
        rel_path = file_path.name

    return {
        "parent": resolved_parent,
        "step": resolved_step,
        "run": resolved_run,
        "rel_path": rel_path,
    }


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


def _file_headers(base: dict[str, str], f: Path, routing: dict[str, str]) -> dict[str, str]:
    meta = _parse_img_meta(f)
    return {
        **base,
        "X-Parent": routing["parent"],
        "X-Step": routing["step"],
        "X-Run": routing["run"],
        "X-Rel-Path": routing["rel_path"],
        "X-Cam": meta["cam"],
        "X-Date": meta["date"],
        "X-Time": meta["time"],
        "X-Type": meta["type"],
    }


def _send_file(
    http,
    server: str,
    base_headers: dict[str, str],
    f: Path,
    routing: dict[str, str],
) -> tuple[bool, float]:
    """Отправить один файл. Возвращает (ok, work_ms)."""
    rel = routing["rel_path"]
    t0 = time.monotonic()
    try:
        data = f.read_bytes()
        if not data:
            return True, 0.0
        headers = _file_headers(base_headers, f, routing)
        resp = http.post(f"{server}/file", content=data, headers=headers)
        resp.raise_for_status()
        ok = resp.json().get("ok", False)
        return ok, (time.monotonic() - t0) * 1000
    except FileNotFoundError:
        return True, (time.monotonic() - t0) * 1000  # удалён между сканом и чтением — нормально
    except Exception as e:
        logger.warning("[!] %s: %s", rel, e)
        return False, (time.monotonic() - t0) * 1000


def cmd_send(args) -> int:
    """Разово отправить все файлы из run_*-каталога на сервер (поштучно)."""
    import httpx

    src = Path(args.src).resolve()
    if not src.is_dir():
        logger.error("[!] Не найдено: %s", src)
        return 1

    step   = src.parent.name
    run    = src.name
    parent = src.parent.parent.name
    server, key = _get_server_and_key(args)
    env    = _load_env()

    exts  = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    files = sorted(f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in exts)

    if not files:
        logger.info("Нет файлов с расширениями %s в %s", exts, src)
        return 0

    base_headers = {"X-Api-Key": key, "Content-Type": "application/octet-stream"}

    logger.info("Источник: %s", src)
    logger.info("Parent:  %s  Шаг: %s  Прогон: %s", parent, step, run)
    logger.info("Сервер:  %s", server)
    logger.info("Файлов:  %d", len(files))

    limiter = _make_limiter(env)
    n_ok = n_fail = 0

    with httpx.Client(timeout=60.0) as http:
        for i, f in enumerate(files, 1):
            routing = _resolve_routing(f, run_root=src, parent=parent, step=step, run=run)
            rel = routing["rel_path"]
            t0 = time.monotonic()
            ok, work_ms = _send_file(http, server, base_headers, f, routing)
            if ok:
                n_ok += 1
                logger.info("[%d/%d] ↑ %s  (%.0fмс)", i, len(files), rel, work_ms)
            else:
                n_fail += 1
            sleep_ms = limiter.sleep(t0)
            msg = limiter.adapt(work_ms, sleep_ms)
            if msg:
                logger.info(msg)

    logger.info("Итого: %d отправлено, %d ошибок", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


def cmd_watch(args) -> int:
    """Постоянно следит за каталогом; новые файлы — отправляет, подтверждённые — удаляет."""
    import httpx

    watch_dir = Path(args.watch_dir).resolve()
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
    base_headers = {"X-Api-Key": key, "Content-Type": "application/octet-stream"}

    if watch_dir.name == "images":
        run_hint = watch_dir.parent.name
        scope_parent = watch_dir.parent.parent.parent.name
        scope_step = watch_dir.parent.parent.name
    else:
        run_hint = "(из пути каждого файла)"
        scope_parent = args.parent or watch_dir.parent.name
        scope_step = args.step or watch_dir.name

    _max_r = f"{1/limiter.min_interval:.1f}" if limiter.min_interval > 0 else "∞"
    _min_r = f"{1/limiter.max_interval:.2f}" if limiter.max_interval > 0 else "0"
    logger.info("=== transfer watch ===")
    logger.info("Каталог: %s", watch_dir)
    logger.info("Parent:  %s  Шаг: %s  Run: %s", scope_parent, scope_step, run_hint)
    logger.info("Сервер:  %s", server)
    logger.info("Расш.:   %s", ', '.join(sorted(exts)))
    logger.info("Скор.:   max=%s/с  min=%s/с  poll=%ss", _max_r, _min_r, poll_sec)

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
                logger.warning("scan: %s", e)
                time.sleep(poll_sec)
                continue

            if new_files:
                logger.info("новый батч: %d файлов", len(new_files))

            now = time.monotonic()
            for _file_idx, f in enumerate(new_files, 1):
                if now < _retry_after.get(f, 0):
                    continue  # backoff not expired yet

                routing = _resolve_routing(
                    f,
                    watch_dir=watch_dir,
                    parent=args.parent,
                    step=args.step,
                    run=args.run,
                )
                rel = routing["rel_path"]
                _rel_parts = f.relative_to(watch_dir).parts if f.is_relative_to(watch_dir) else ()
                _date_part = next(
                    (p for p in _rel_parts if len(p) == 8 and p.isdigit()),
                    "",
                )

                t0 = time.monotonic()
                try:
                    data = f.read_bytes()
                    if not data:
                        continue
                    headers = _file_headers(base_headers, f, routing)
                    resp = http.post(f"{server}/file", content=data, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()
                    work_ms = (time.monotonic() - t0) * 1000
                    if result.get("ok"):
                        f.unlink(missing_ok=True)
                        _fail_count.pop(f, None)
                        _retry_after.pop(f, None)
                        n_sent += 1
                        _run_tag = f"  run={routing['run']}" if routing["run"] else ""
                        _date_tag = f"  [дата {_date_part}]" if _date_part else ""
                        logger.info("[в батче %d/%d] ↑ %s  (%.1f КБ  %.0fмс)  всего=%d%s%s",
                                    _file_idx, len(new_files), rel,
                                    len(data) / 1024, work_ms, n_sent, _run_tag, _date_tag)
                    else:
                        n_failed += 1
                        work_ms = (time.monotonic() - t0) * 1000
                        logger.warning("[!] нет подтверждения: %s", rel)
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
                    logger.warning("[!] %s: %s  (retry in %.0fs)", rel, e, delay)

                sleep_ms = limiter.sleep(t0)
                msg = limiter.adapt(work_ms, sleep_ms)
                if msg:
                    logger.info(msg)

            if not new_files:
                _sn = Path(sys.argv[0]).stem
                logger.info("(%s) Файлов нет в %s — ожидание %.0fs…", _sn, watch_dir, poll_sec)
                time.sleep(poll_sec)


def cmd_health(args) -> int:
    import httpx
    server, _ = _get_server_and_key(args)
    try:
        r = httpx.get(f"{server}/health", timeout=5)
        r.raise_for_status()
        logger.info("%s  →  %s", server, r.json())
        return 0
    except Exception as e:
        logger.error("[!] %s", e)
        return 1


def cmd_runs(args) -> int:
    import httpx
    server, key = _get_server_and_key(args)
    r = httpx.get(f"{server}/runs", headers={"X-Api-Key": key}, timeout=10)
    r.raise_for_status()
    data: dict = r.json()
    if not data:
        logger.info("(нет принятых прогонов)")
        return 0
    for step, runs in sorted(data.items()):
        logger.info("%s:", step)
        for run in runs:
            logger.info("  %s", run)
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
    p_watch.add_argument("watch_dir", help="Каталог наблюдения (run_XXX/images или 1_motion_diff/)")
    p_watch.add_argument("--run-root", default="",
                         help="(устар.) не используется — см. --parent/--step/--run")
    p_watch.add_argument("--parent",   default="",
                         help="Parent (default: parent каталога watch_dir, напр. pipeline)")
    p_watch.add_argument("--step",     default="",
                         help="Шаг (default: имя watch_dir или 1_motion_diff)")
    p_watch.add_argument("--run",      default="",
                         help="Прогон (default: run_XXX из пути или каталога images)")
    p_watch.add_argument("--ext",      default="jpg", metavar="EXTS",
                         help="Расширения файлов через запятую (default: jpg)")

    sub.add_parser("health", help="Проверить доступность сервера")
    sub.add_parser("runs",   help="Список принятых прогонов на сервере")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 1
    setup_logging()

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
