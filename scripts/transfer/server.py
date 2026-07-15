"""Сервер приёма run_*-каталогов от клиентских машин.

Клиент упаковывает run_*-каталог в tar.gz и отправляет POST-запросом.
Сервер распаковывает в .output/pipeline/<step>/.

Usage:
    python scripts/transfer/server.py
    python scripts/transfer/server.py --port 8765 --output .output/pipeline
    TRANSFER_API_KEY=secret uvicorn scripts.transfer.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from common.utils.access import is_ip_allowed, log_ip_denied, parse_allowed_ips
from common.utils.time_msk import MSK, ts_iso

logger = logging.getLogger(__name__)

API_KEY: str = os.environ.get("TRANSFER_API_KEY", "")
ALLOWED_IPS = parse_allowed_ips(os.environ.get("ALLOWED_IPS", ""))
OUTPUT_DIR: Path = REPO_ROOT / ".output" / "transfer"

app = FastAPI(title="Transfer Server", version="1.0")


class _IPAllowlistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else None
        if not is_ip_allowed(client, ALLOWED_IPS):
            key = request.headers.get("x-api-key", "")
            log_ip_denied(
                logger,
                client_ip=client,
                service="Transfer",
                path=request.url.path,
                api_key_valid=bool(API_KEY and key == API_KEY),
            )
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        return await call_next(request)


if ALLOWED_IPS is not None:
    app.add_middleware(_IPAllowlistMiddleware)


def _check_key(key: str) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    resolved = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(resolved)):
            raise ValueError(f"Path traversal in archive: {member.name}")
    tar.extractall(dest)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/pipeline/{step}")
async def receive_run(
    step: str,
    request: Request,
    x_api_key: str = Header(default=""),
) -> JSONResponse:
    """Принять tar.gz с run_*-каталогом и распаковать в .output/pipeline/<step>/."""
    _check_key(x_api_key)

    step_dir = OUTPUT_DIR / step
    step_dir.mkdir(parents=True, exist_ok=True)

    # Записываем тело запроса во временный файл (поддержка больших архивов)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tmp = Path(tf.name)
        async for chunk in request.stream():
            tf.write(chunk)

    try:
        with tarfile.open(tmp, "r:gz") as tar:
            top_dirs = {
                Path(m.name).parts[0]
                for m in tar.getmembers()
                if m.name and not m.name.startswith("/")
            }
            try:
                _safe_extractall(tar, step_dir)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)

    run_name = next(iter(top_dirs)) if len(top_dirs) == 1 else ""
    dest = str(step_dir / run_name) if run_name else str(step_dir)
    print(f"  [transfer] {step}/{run_name}  →  {dest}")
    return JSONResponse({"ok": True, "step": step, "run": run_name, "dest": dest})


def _validate_path_component(value: str, name: str) -> None:
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(400, f"Invalid header {name}: {value!r}")


def _captured_at_iso(date: str, time_str: str) -> str | None:
    """Собрать ISO 8601 из date=YYYYMMDD и time=HHMMSS[_ffffff] (MSK)."""
    if not date or len(date) != 8 or not date.isdigit():
        return None
    if not time_str:
        return None
    parts = time_str.split("_")
    hms = parts[0]
    micro = parts[1] if len(parts) > 1 else "000000"
    if len(hms) != 6 or not hms.isdigit():
        return None
    try:
        dt = datetime(
            int(date[:4]), int(date[4:6]), int(date[6:8]),
            int(hms[:2]), int(hms[2:4]), int(hms[4:6]),
            int(micro[:6].ljust(6, "0")[:6]),
            tzinfo=MSK,
        )
        return dt.isoformat()
    except ValueError:
        return None


def _write_meta_sidecar(
    dest_file: Path,
    *,
    parent: str,
    step: str,
    run: str,
    rel_path: str,
    cam: str,
    date: str,
    time_str: str,
    img_type: str,
    size_bytes: int,
    meta_layout: str,
) -> Path:
    """Записать sidecar JSON рядом с diff/service или в legacy-ветке meta/."""
    captured_at = _captured_at_iso(date, time_str)
    payload = {
        "parent": parent,
        "step": step,
        "run": run,
        "cam": cam,
        "date": date,
        "time": time_str,
        "timezone": "Europe/Moscow",
        "captured_at": captured_at,
        "type": img_type,
        "rel_path": rel_path,
        "received_at": ts_iso(),
        "size_bytes": size_bytes,
        "dest_image": str(dest_file),
    }

    stem = dest_file.stem
    if meta_layout == "dated":
        meta_file = (OUTPUT_DIR / "meta" / date / cam / f"{stem}.json").resolve()
    else:
        meta_file = (
            OUTPUT_DIR / "meta" / "_legacy" / step / run / f"{stem}.json"
        ).resolve()

    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_file


@app.post("/file")
async def receive_file(
    request: Request,
    x_api_key: str = Header(default=""),
    x_parent: str = Header(default=""),
    x_step: str = Header(default=""),
    x_run: str = Header(default=""),
    x_rel_path: str = Header(default=""),
    x_cam: str = Header(default=""),
    x_date: str = Header(default=""),
    x_time: str = Header(default=""),
    x_type: str = Header(default=""),
) -> JSONResponse:
    """Принять один файл.

    Если переданы X-Cam / X-Date / X-Type — раскладывает по структуре:
      diff/YYYYMMDD/<cam>/   — тип diff
      service/YYYYMMDD/<cam>/ — baseline, heartbeat и прочие
      meta/YYYYMMDD/<cam>/ — sidecar JSON с parent/step/run и временами

    Иначе (legacy) — OUTPUT_DIR/{step}/{run}/{rel_path} + meta/_legacy/{step}/{run}/.
    """
    _check_key(x_api_key)

    filename = Path(x_rel_path).name if x_rel_path else ""
    meta_layout = "legacy"

    if x_cam and x_date and filename:
        _validate_path_component(x_cam, "X-Cam")
        _validate_path_component(x_date, "X-Date")
        if not x_date.isdigit() or len(x_date) != 8:
            raise HTTPException(400, f"Invalid X-Date: {x_date!r}")

        subdir = "diff" if x_type == "diff" else "service"
        dest_dir = OUTPUT_DIR / subdir / x_date / x_cam
        dest_file = (dest_dir / filename).resolve()
        try:
            dest_file.relative_to((OUTPUT_DIR / subdir).resolve())
        except ValueError:
            raise HTTPException(400, "Path traversal detected")
        log_tag = f"{subdir}/{x_date}/{x_cam}/{filename}"
        meta_layout = "dated"
    else:
        if not x_step or not x_run or not x_rel_path:
            raise HTTPException(400, "X-Cam+X-Date or X-Step+X-Run+X-Rel-Path headers required")
        for part in (x_step, x_run):
            _validate_path_component(part, "X-Step/X-Run")
        run_dir   = (OUTPUT_DIR / x_step / x_run).resolve()
        dest_file = (run_dir / x_rel_path).resolve()
        try:
            dest_file.relative_to(run_dir)
        except ValueError:
            raise HTTPException(400, "Path traversal detected")
        log_tag = f"{x_step}/{x_run}/{x_rel_path}"

    dest_file.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    tmp_file = dest_file.with_suffix(".tmp")
    size = 0
    with open(tmp_file, "wb") as fh:
        async for chunk in request.stream():
            fh.write(chunk)
            size += len(chunk)
    tmp_file.rename(dest_file)  # атомарный rename — файл появляется целиком

    meta_file = _write_meta_sidecar(
        dest_file,
        parent=x_parent,
        step=x_step,
        run=x_run,
        rel_path=x_rel_path,
        cam=x_cam,
        date=x_date,
        time_str=x_time,
        img_type=x_type,
        size_bytes=size,
        meta_layout=meta_layout,
    )
    elapsed = time.monotonic() - t0

    logger.info("[recv] %s  (%.1f КБ)  Готово. Время: %.2f с.  meta=%s",
                log_tag, size / 1024, elapsed, meta_file.name)
    return JSONResponse({"ok": True, "dest": str(dest_file), "meta": str(meta_file)})


@app.get("/runs")
def list_runs() -> dict:
    """Список всех принятых run_* по шагам."""
    result: dict[str, list[str]] = {}
    if OUTPUT_DIR.is_dir():
        for step_dir in sorted(OUTPUT_DIR.iterdir()):
            if step_dir.is_dir():
                result[step_dir.name] = sorted(
                    r.name for r in step_dir.iterdir() if r.is_dir()
                )
    return result


def _log_config() -> dict:
    """Uvicorn log config with timestamps on all lines."""
    import copy
    from uvicorn.config import LOGGING_CONFIG
    cfg = copy.deepcopy(LOGGING_CONFIG)

    cfg["formatters"]["default"] = {
        "format":  "%(asctime)s  %(levelname)-8s  %(message)s",
        "datefmt": "%H:%M:%S",
    }
    cfg["formatters"]["access"] = {
        "()":       "uvicorn.logging.AccessFormatter",
        "fmt":      '%(asctime)s  ACCESS    %(client_addr)s - "%(request_line)s" %(status_code)s',
        "datefmt":  "%H:%M:%S",
        "use_colors": False,
    }
    cfg["loggers"][""] = {"handlers": ["default"], "level": "INFO"}
    return cfg


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Transfer server")
    ap.add_argument("--host",   default=os.environ.get("TRANSFER_HOST", "0.0.0.0"))
    ap.add_argument("--port",   type=int, default=int(os.environ.get("TRANSFER_PORT", "8765")))
    ap.add_argument("--output", type=Path, default=None,
                    help="Корень для распаковки (default: .output/pipeline)")
    args = ap.parse_args()

    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output.resolve()

    if not API_KEY:
        logger.warning("[!] TRANSFER_API_KEY не задан — сервер открыт без аутентификации")

    logger.info("Transfer server: http://%s:%s", args.host, args.port)
    logger.info("Output dir:      %s", OUTPUT_DIR)
    if ALLOWED_IPS is not None:
        logger.info("Allowed IPs:     %s", ALLOWED_IPS.summary())
    else:
        logger.warning("[!] ALLOWED_IPS не задан — доступ с любого IP")
    uvicorn.run(app, host=args.host, port=args.port, log_config=_log_config())


if __name__ == "__main__":
    main()
