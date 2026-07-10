"""Тесты collect_scenes.py — сбор кропов по сценам."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "train"))

import collect_scenes as cs


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_jpg(path: Path, bgr: tuple[int, int, int] = (128, 128, 128)) -> None:
    img = np.full((64, 64, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _frame_name(cam: str, date: str, hh: int, mm: int, ss: int, conf: float = 0.80) -> str:
    return (
        f"{cam}_9_d_{date}_{hh:02d}{mm:02d}{ss:02d}_000000"
        f"_0000_sometoken_p1of1_conf{conf:.2f}.jpg"
    )


# ── unit: _parse_frame ───────────────────────────────────────────────────────

def test_parse_frame_valid():
    name = _frame_name("cam_01", "20260706", 9, 12, 34)
    result = cs._parse_frame(Path(name).stem)
    assert result is not None
    cam, date, sod, conf = result
    assert cam == "cam_01"
    assert date == "20260706"
    assert sod == 9 * 3600 + 12 * 60 + 34
    assert abs(conf - 0.80) < 0.01


def test_parse_frame_invalid():
    assert cs._parse_frame("random_file_name") is None


def test_parse_frame_conf():
    name = _frame_name("cam_02", "20260101", 0, 0, 0, conf=0.92)
    _, _, _, conf = cs._parse_frame(Path(name).stem)
    assert abs(conf - 0.92) < 0.01


# ── unit: _persons_in_frame ──────────────────────────────────────────────────

def test_persons_in_frame_single():
    assert cs._persons_in_frame("cam_01_9_d_20260706_123456_000_p1of1_conf0.90") == 1


def test_persons_in_frame_two():
    assert cs._persons_in_frame("cam_01_9_d_20260706_123456_000_p2of2_conf0.90") == 2


def test_persons_in_frame_no_tag():
    assert cs._persons_in_frame("cam_01_9_d_20260706_123456_conf0.90") == 1


# ── unit: _blur_score ────────────────────────────────────────────────────────

def test_blur_score_sharp(tmp_path):
    p = tmp_path / "sharp.jpg"
    # Checkerboard — very sharp
    img = np.zeros((64, 64), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(p), bgr)
    assert cs._blur_score(p) > 5000


def test_blur_score_blurry(tmp_path):
    p = tmp_path / "blurry.jpg"
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    assert cs._blur_score(p) < 10


def test_blur_score_missing(tmp_path):
    assert cs._blur_score(tmp_path / "nonexistent.jpg") == 0.0


# ── unit: _mean_diff ─────────────────────────────────────────────────────────

def test_mean_diff_identical(tmp_path):
    a = tmp_path / "a.jpg"
    _make_jpg(a, (100, 100, 100))
    d = cs._mean_diff(a, a)
    assert d < 5.0  # very small (JPEG round-trip)


def test_mean_diff_different(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _make_jpg(a, (0, 0, 0))
    _make_jpg(b, (255, 255, 255))
    d = cs._mean_diff(a, b)
    assert d > 200


def test_mean_diff_missing(tmp_path):
    a = tmp_path / "a.jpg"
    _make_jpg(a)
    d = cs._mean_diff(a, tmp_path / "missing.jpg")
    assert d == 0.0


# ── unit: _load_labels ───────────────────────────────────────────────────────

def test_load_labels_names(tmp_path):
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({
        "/abs/path/file1.jpg": "1_resident",
        "file2.jpg": "4_guest",
    }), encoding="utf-8")
    labels = cs._load_labels(tmp_path)
    assert labels == {"file1.jpg": "1_resident", "file2.jpg": "4_guest"}


def test_load_labels_nested(tmp_path):
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({
        "labels": {"/some/path/img.jpg": "1_resident"}
    }), encoding="utf-8")
    labels = cs._load_labels(tmp_path)
    assert labels == {"img.jpg": "1_resident"}


def test_load_labels_missing(tmp_path):
    assert cs._load_labels(tmp_path) == {}


# ── integration: collect_scenes (temporal gap) ───────────────────────────────

def _make_source_dir(parent: Path, date: str, cls: str,
                     frames: list[tuple[int, int, int]]) -> Path:
    """Создаёт src_dir с JPG-файлами. frames = [(hh, mm, ss), ...]."""
    src = parent / date / cls
    src.mkdir(parents=True)
    for i, (hh, mm, ss) in enumerate(frames):
        name = _frame_name("cam_01", date, hh, mm, ss, conf=0.85)
        _make_jpg(src / name, bgr=(i * 30 % 255, 128, 64))
    return src


def test_collect_scenes_temporal_gap(tmp_path):
    """Два блока кадров с разрывом > scene_gap → 2 сцены."""
    src = _make_source_dir(
        tmp_path / "src", "20260706", "1_resident",
        [(9, 0, 0), (9, 0, 5), (9, 0, 10),    # сцена 1
         (9, 10, 0), (9, 10, 5)],              # сцена 2 (gap = 590s > 120s)
    )
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=0,
        min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
    )
    scenes_data = json.loads((out / "scenes.json").read_text())
    assert len(scenes_data["scenes"]) == 2
    assert sum(len(s["frames"]) for s in scenes_data["scenes"]) == 5


def test_collect_scenes_single_scene(tmp_path):
    """Кадры без разрыва → 1 сцена."""
    src = _make_source_dir(
        tmp_path / "src", "20260706", "1_resident",
        [(9, 0, 0), (9, 0, 3), (9, 0, 6)],
    )
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=0,
        min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
    )
    scenes_data = json.loads((out / "scenes.json").read_text())
    assert len(scenes_data["scenes"]) == 1
    assert len(scenes_data["scenes"][0]["frames"]) == 3


def test_collect_scenes_representative(tmp_path):
    """Каждая сцена имеет поле representative и соответствующий файл в representatives/."""
    src = _make_source_dir(
        tmp_path / "src", "20260706", "1_resident",
        [(9, 0, 0), (9, 0, 5)],
    )
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=0,
        min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
    )
    scenes_data = json.loads((out / "scenes.json").read_text())
    for scene in scenes_data["scenes"]:
        assert "representative" in scene
        rep_name = scene["representative"]
        assert rep_name in scene["frames"]
        # репрезентативный кадр должен быть скопирован
        rep_file = list((out / "representatives").glob(f"*{rep_name}"))
        assert rep_file, f"representative {rep_name} not found in representatives/"


def test_collect_scenes_max_persons_filter(tmp_path):
    """Кадры с _p2of2_ должны быть пропущены при max_persons=1."""
    src = tmp_path / "20260706" / "1_resident"
    src.mkdir(parents=True)
    good = _frame_name("cam_01", "20260706", 9, 0, 0)
    bad = good.replace("p1of1", "p2of2")
    _make_jpg(src / good)
    _make_jpg(src / bad)
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=0, min_blur=0, min_conf=0,
        max_persons=1,
        scene_gap_sec=120, scene_diff=0,
    )
    scenes_data = json.loads((out / "scenes.json").read_text())
    all_frames = [n for s in scenes_data["scenes"] for n in s["frames"]]
    assert good in all_frames
    assert bad not in all_frames


def test_collect_scenes_sibling_labels(tmp_path):
    """labels.json должен включить кадры из соседнего класса в collect_classes."""
    date = "20260706"
    date_dir = tmp_path / "src" / date
    # Основной класс — 2_delivery (источник)
    delivery = date_dir / "2_delivery"
    delivery.mkdir(parents=True)
    name_delivery = _frame_name("cam_01", date, 9, 0, 0)
    _make_jpg(delivery / name_delivery)
    # Соседний класс — 3_utilities, переразмечен в 1_resident через labels.json
    utilities = date_dir / "3_utilities"
    utilities.mkdir()
    name_util = _frame_name("cam_01", date, 9, 1, 0)
    _make_jpg(utilities / name_util)
    lf = date_dir / "labels.json"
    lf.write_text(json.dumps({name_util: "1_resident"}), encoding="utf-8")

    out = tmp_path / "pool"
    cs.collect_scenes(
        [delivery], out,
        collect_classes={"1_resident", "2_delivery"},
        interval=0, min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
    )
    all_frames = [n for s in json.loads((out / "scenes.json").read_text())["scenes"]
                  for n in s["frames"]]
    assert name_delivery in all_frames
    assert name_util in all_frames


def test_collect_scenes_dry_run(tmp_path):
    """dry_run не должен создавать файлы в out_dir."""
    src = _make_source_dir(
        tmp_path / "src", "20260706", "1_resident",
        [(9, 0, 0)],
    )
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=0, min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
        dry_run=True,
    )
    assert not (out / "scenes.json").exists()
    assert not (out / "representatives").exists()


def test_collect_scenes_interval_dedup(tmp_path):
    """interval=5: кадры в одном окне дедуплицируются — остаётся только 1."""
    src = tmp_path / "20260706" / "1_resident"
    src.mkdir(parents=True)
    # 3 кадра в окне 0–4с (все в bucket sod//5==0), conf разный
    for ss, conf in [(0, 0.70), (1, 0.85), (3, 0.75)]:
        name = _frame_name("cam_01", "20260706", 9, 0, ss, conf=conf)
        _make_jpg(src / name)
    out = tmp_path / "pool"
    cs.collect_scenes(
        [src], out,
        collect_classes={"1_resident"},
        interval=5, min_blur=0, min_conf=0,
        scene_gap_sec=120, scene_diff=0,
    )
    scenes_data = json.loads((out / "scenes.json").read_text())
    all_frames = [n for s in scenes_data["scenes"] for n in s["frames"]]
    assert len(all_frames) == 1
    # должен остаться кадр с наибольшим conf=0.85
    assert "conf0.85" in all_frames[0]
