"""Тесты propagate_scenes.py — авторазметка кадров внутри сцен."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "train"))

import propagate_scenes as ps


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_jpg(path: Path, bgr=(128, 128, 128)):
    img = np.full((32, 32, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_pool(tmp_path: Path, scenes: list[list[str]]) -> Path:
    """Создаёт scene_pool с несколькими сценами.

    scenes — список сцен; каждая сцена — список имён файлов.
    Возвращает путь к pool_dir.
    """
    pool = tmp_path / "pool"
    pool.mkdir()
    scene_data = []
    for sid, names in enumerate(scenes):
        for name in names:
            p = pool / name
            if not p.exists():
                _make_jpg(p)
        scene_data.append({
            "scene_id": sid,
            "cam": "cam_01",
            "date": "20260706",
            "start_time": sid * 600,
            "frames": names,
            "representative": names[0],
            "representative_blur": 100.0,
            "split_reason": "",
        })
    (pool / "scenes.json").write_text(
        json.dumps({"scene_gap_sec": 120, "scene_diff": 50.0, "scenes": scene_data}),
        encoding="utf-8",
    )
    return pool


def _write_labels(pool: Path, labels: dict[str, str]) -> None:
    existing = {}
    lf = pool / "labels.json"
    if lf.exists():
        existing = json.loads(lf.read_text())
    existing.update(labels)
    lf.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")


def _read_labels(pool: Path) -> dict[str, str]:
    lf = pool / "labels.json"
    if not lf.exists():
        return {}
    raw = json.loads(lf.read_text())
    return {Path(k).name: v for k, v in raw.get("labels", raw).items()}


# ── unit: _load_labels ───────────────────────────────────────────────────────

def test_load_labels_flat(tmp_path):
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({"frame.jpg": "p1", "/abs/path/x.jpg": "p2"}))
    out = ps._load_labels(tmp_path)
    assert out == {"frame.jpg": "p1", "x.jpg": "p2"}


def test_load_labels_nested(tmp_path):
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({"labels": {"a.jpg": "p1"}}))
    assert ps._load_labels(tmp_path) == {"a.jpg": "p1"}


def test_load_labels_skips_skip(tmp_path):
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({"a.jpg": "skip", "b.jpg": "p1"}))
    labels = ps._load_labels(tmp_path)
    # "skip" значения тоже загружаются — фильтрует propagate()
    assert "b.jpg" in labels


def test_load_labels_missing(tmp_path):
    assert ps._load_labels(tmp_path) == {}


# ── integration: propagate ───────────────────────────────────────────────────

def test_propagate_basic(tmp_path):
    """Одна размеченная сцена из 3 кадров → 2 должны быть авторазмечены."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg", "c.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01"})
    result = ps.propagate(pool)
    assert result["propagated"] == 2
    labels = _read_labels(pool)
    assert labels["b.jpg"] == "person_01"
    assert labels["c.jpg"] == "person_01"


def test_propagate_dry_run(tmp_path):
    """dry_run не должен менять labels.json."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01"})
    ps.propagate(pool, dry_run=True)
    labels = _read_labels(pool)
    assert "b.jpg" not in labels


def test_propagate_multi_scene(tmp_path):
    """Две сцены с разными людьми → метки не смешиваются."""
    pool = _make_pool(tmp_path, [
        ["s0_a.jpg", "s0_b.jpg"],
        ["s1_a.jpg", "s1_b.jpg"],
    ])
    _write_labels(pool, {"s0_a.jpg": "person_01", "s1_a.jpg": "person_02"})
    ps.propagate(pool)
    labels = _read_labels(pool)
    assert labels["s0_b.jpg"] == "person_01"
    assert labels["s1_b.jpg"] == "person_02"


def test_propagate_conflict_not_written(tmp_path):
    """Сцена с двумя разными метками → не распространяется (конфликт)."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg", "c.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01", "b.jpg": "person_02"})
    result = ps.propagate(pool)
    assert result["conflicts"] == 1
    labels = _read_labels(pool)
    assert "c.jpg" not in labels


def test_propagate_unlabeled_scene_skipped(tmp_path):
    """Сцена без ни одной ручной метки → полностью пропускается."""
    pool = _make_pool(tmp_path, [
        ["labeled_a.jpg", "labeled_b.jpg"],
        ["unlabeled_c.jpg", "unlabeled_d.jpg"],
    ])
    _write_labels(pool, {"labeled_a.jpg": "person_01"})
    result = ps.propagate(pool)
    assert result["unlabeled_scenes"] == 1
    labels = _read_labels(pool)
    assert "unlabeled_c.jpg" not in labels
    assert "unlabeled_d.jpg" not in labels
    assert labels.get("labeled_b.jpg") == "person_01"


def test_propagate_already_labeled_counted(tmp_path):
    """Кадры, уже имеющие метку, считаются в n_already, не в propagated."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg", "c.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01", "b.jpg": "person_01"})
    result = ps.propagate(pool)
    assert result["propagated"] == 1   # только c.jpg
    # b.jpg не перезаписывается (уже было)
    labels = _read_labels(pool)
    assert labels["b.jpg"] == "person_01"


def test_propagate_skip_label_ignored(tmp_path):
    """Метка 'skip' не должна распространяться как person_id."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"]])
    _write_labels(pool, {"a.jpg": "skip"})
    result = ps.propagate(pool)
    # Сцена без валидного person_id → unlabeled
    assert result["unlabeled_scenes"] == 1
    assert result["propagated"] == 0


def test_propagate_intra_diff_limit(tmp_path):
    """intra_diff_limit=0 (выкл.) не должен отбрасывать кадры."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01"})
    result = ps.propagate(pool, intra_diff_limit=0.0)
    assert result["diff_rejected"] == 0
    assert result["propagated"] == 1


def test_propagate_intra_diff_limit_reject(tmp_path):
    """intra_diff_limit очень маленький → почти всё отбрасывается."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"]])
    # a.jpg — чёрный, b.jpg — белый → diff ~ 255
    _make_jpg(pool / "a.jpg", (0, 0, 0))
    _make_jpg(pool / "b.jpg", (255, 255, 255))
    _write_labels(pool, {"a.jpg": "person_01"})
    result = ps.propagate(pool, intra_diff_limit=10.0)
    assert result["diff_rejected"] == 1
    assert result["propagated"] == 0


def test_propagate_no_scenes_json(tmp_path):
    """Отсутствие scenes.json → пустой результат без ошибки."""
    pool = tmp_path / "pool"
    pool.mkdir()
    result = ps.propagate(pool)
    assert result == {}


def test_propagate_scene_prefix_stripped(tmp_path):
    """Метки из representatives/ (с префиксом scene0042_) должны совпадать с кадрами сцены."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"]])
    # label_ui сохранит имя с префиксом — как в representatives/
    _write_labels(pool, {"scene0000_a.jpg": "person_01"})
    result = ps.propagate(pool)
    assert result["propagated"] == 1
    labels = _read_labels(pool)
    assert labels.get("b.jpg") == "person_01"


def test_propagate_labels_preserved(tmp_path):
    """Существующие метки в labels.json должны быть сохранены при обновлении."""
    pool = _make_pool(tmp_path, [["a.jpg", "b.jpg"], ["c.jpg", "d.jpg"]])
    _write_labels(pool, {"a.jpg": "person_01", "c.jpg": "person_02"})
    ps.propagate(pool)
    labels = _read_labels(pool)
    # Исходные метки должны сохраниться
    assert labels.get("a.jpg") == "person_01"
    assert labels.get("c.jpg") == "person_02"
    # Новые должны быть добавлены
    assert labels.get("b.jpg") == "person_01"
    assert labels.get("d.jpg") == "person_02"
