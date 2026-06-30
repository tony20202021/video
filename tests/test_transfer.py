"""Тесты Transfer Server (server.py) и клиентских утилит (client.py)."""

from __future__ import annotations

import tarfile
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "pipeline"


@pytest.fixture()
def app_client(output_dir: Path, monkeypatch):
    """TestClient с переопределённым OUTPUT_DIR и API_KEY."""
    import scripts.transfer.server as srv

    monkeypatch.setattr(srv, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(srv, "API_KEY", "testkey")
    return TestClient(srv.app)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Минимальный run_*-каталог для тестов."""
    step_dir = tmp_path / "1_motion_diff"
    run = step_dir / "run_20260101_120000_msk"
    (run / "images").mkdir(parents=True)
    (run / "frames.csv").write_text("ts,path\n0,img.jpg\n", encoding="utf-8")
    (run / "images" / "img001.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    return run


def _make_tarball(run: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(run, arcname=run.name)
    return buf.getvalue()


# ─── Server: health ───────────────────────────────────────────────────────────

def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ─── Server: auth ────────────────────────────────────────────────────────────

def test_receive_wrong_key(app_client, run_dir):
    tarball = _make_tarball(run_dir)
    r = app_client.post(
        "/pipeline/1_motion_diff",
        content=tarball,
        headers={"X-Api-Key": "wrong", "Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 401


def test_receive_no_key_when_required(app_client, run_dir):
    tarball = _make_tarball(run_dir)
    r = app_client.post(
        "/pipeline/1_motion_diff",
        content=tarball,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 401


# ─── Server: receive ─────────────────────────────────────────────────────────

def test_receive_extracts_run(app_client, run_dir, output_dir):
    tarball = _make_tarball(run_dir)
    r = app_client.post(
        "/pipeline/1_motion_diff",
        content=tarball,
        headers={"X-Api-Key": "testkey", "Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["step"] == "1_motion_diff"
    assert body["run"] == run_dir.name

    dest = output_dir / "1_motion_diff" / run_dir.name
    assert dest.is_dir()
    assert (dest / "frames.csv").is_file()
    assert (dest / "images" / "img001.jpg").is_file()


def test_receive_idempotent(app_client, run_dir, output_dir):
    """Повторная отправка того же run_* перезаписывает, не падает."""
    tarball = _make_tarball(run_dir)
    for _ in range(2):
        r = app_client.post(
            "/pipeline/1_motion_diff",
            content=tarball,
            headers={"X-Api-Key": "testkey", "Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200


# ─── Server: list runs ────────────────────────────────────────────────────────

def test_list_runs_empty(app_client):
    r = app_client.get("/runs")
    assert r.status_code == 200
    assert r.json() == {}


def test_list_runs_after_upload(app_client, run_dir):
    tarball = _make_tarball(run_dir)
    app_client.post(
        "/pipeline/1_motion_diff",
        content=tarball,
        headers={"X-Api-Key": "testkey", "Content-Type": "application/octet-stream"},
    )
    r = app_client.get("/runs")
    assert r.status_code == 200
    data = r.json()
    assert "1_motion_diff" in data
    assert run_dir.name in data["1_motion_diff"]


# ─── Server: path traversal guard ────────────────────────────────────────────

def test_path_traversal_rejected(app_client, tmp_path):
    """Архив с ../evil не должен быть извлечён."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="../evil.txt")
        data = b"pwned"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    r = app_client.post(
        "/pipeline/1_motion_diff",
        content=buf.getvalue(),
        headers={"X-Api-Key": "testkey", "Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 400  # path traversal отклонён


# ─── Client: pack / step detection ───────────────────────────────────────────

def test_pack_creates_valid_tarball(run_dir, tmp_path):
    from scripts.transfer.client import _pack

    tmp = _pack(run_dir)
    try:
        assert tmp.is_file()
        assert tmp.stat().st_size > 0
        with tarfile.open(tmp, "r:gz") as tf:
            names = tf.getnames()
        assert any(n.startswith(run_dir.name) for n in names)
    finally:
        tmp.unlink(missing_ok=True)


def test_step_inferred_from_parent(run_dir):
    """Шаг определяется из имени родительского каталога run_*."""
    step = run_dir.parent.name
    assert step == "1_motion_diff"
