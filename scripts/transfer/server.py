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

API_KEY: str = os.environ.get("TRANSFER_API_KEY", "")
OUTPUT_DIR: Path = REPO_ROOT / ".output" / "pipeline"

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


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Transfer server")
    ap.add_argument("--host",   default="0.0.0.0")
    ap.add_argument("--port",   type=int, default=8765)
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
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
