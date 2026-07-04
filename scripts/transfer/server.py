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
import os
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

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


_SERVER_ENV = _load_env()
API_KEY: str = os.environ.get("TRANSFER_API_KEY") or _SERVER_ENV.get("TRANSFER_API_KEY", "")
OUTPUT_DIR: Path = REPO_ROOT / ".output" / "transfer"

app = FastAPI(title="Transfer Server", version="1.0")


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


@app.post("/file")
async def receive_file(
    request: Request,
    x_api_key: str = Header(default=""),
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

    Иначе (legacy) — OUTPUT_DIR/{step}/{run}/{rel_path}.
    """
    _check_key(x_api_key)

    filename = Path(x_rel_path).name if x_rel_path else ""

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

    with open(dest_file, "wb") as fh:
        async for chunk in request.stream():
            fh.write(chunk)

    size = dest_file.stat().st_size
    logger.info("[recv] %s  (%.1f КБ)", log_tag, size / 1024)
    return JSONResponse({"ok": True, "dest": str(dest_file)})


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
    """Uvicorn log config extended to route app-level logs through the default handler."""
    import copy
    from uvicorn.config import LOGGING_CONFIG
    cfg = copy.deepcopy(LOGGING_CONFIG)
    # Root logger → default handler so our logger.info/warning() appear alongside uvicorn lines
    cfg["loggers"][""] = {"handlers": ["default"], "level": "INFO"}
    return cfg


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Transfer server")
    ap.add_argument("--host",   default=_SERVER_ENV.get("TRANSFER_HOST", "0.0.0.0"))
    ap.add_argument("--port",   type=int, default=int(_SERVER_ENV.get("TRANSFER_PORT", "8765")))
    ap.add_argument("--output", type=Path, default=None,
                    help="Корень для распаковки (default: .output/pipeline)")
    args = ap.parse_args()

    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output.resolve()

    if not API_KEY:
        print("[!] TRANSFER_API_KEY не задан — сервер открыт без аутентификации",
              file=sys.stderr)

    print(f"Transfer server: http://{args.host}:{args.port}")
    print(f"Output dir:      {OUTPUT_DIR}")
    uvicorn.run(app, host=args.host, port=args.port, log_config=_log_config())


if __name__ == "__main__":
    main()
