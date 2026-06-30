"""Тесты трекера и классификатора направления движения."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.pipeline.tracker import IoUTracker, Track, _iou
from scripts.pipeline.track_direction import (
    classify_direction,
    load_detections,
    run_tracking,
    DIRECTION_HOME,
    DIRECTION_AWAY,
    DIRECTION_UNKNOWN,
    DEFAULT_DOOR_ZONE_X,
    DEFAULT_ELEVATOR_ZONE_X,
)


# ─── IoU ─────────────────────────────────────────────────────────────────────

def test_iou_overlapping():
    a = [0.0, 0.0, 2.0, 2.0]
    b = [1.0, 1.0, 3.0, 3.0]
    assert abs(_iou(a, b) - 1 / 7) < 1e-6


def test_iou_identical():
    box = [10.0, 10.0, 20.0, 20.0]
    assert _iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert _iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


# ─── Tracker ─────────────────────────────────────────────────────────────────

def test_tracker_single_track():
    tracker = IoUTracker(iou_threshold=0.3, max_age=2, min_hits=2)
    tracker.update("f001.jpg", [[100, 100, 200, 300]])
    tracker.update("f002.jpg", [[105, 105, 205, 305]])
    tracker.update("f003.jpg", [[110, 108, 210, 308]])
    tracker.flush()
    tracks = tracker.all_confirmed_tracks()
    assert len(tracks) == 1
    assert tracks[0].hits == 3


def test_tracker_two_people():
    tracker = IoUTracker(iou_threshold=0.3, max_age=2, min_hits=2)
    tracker.update("f001.jpg", [[10, 100, 60, 300], [500, 100, 550, 300]])
    tracker.update("f002.jpg", [[12, 102, 62, 302], [502, 102, 552, 302]])
    tracker.flush()
    assert len(tracker.all_confirmed_tracks()) == 2


def test_tracker_gap_within_max_age():
    tracker = IoUTracker(iou_threshold=0.3, max_age=3, min_hits=2)
    tracker.update("f001.jpg", [[100, 100, 200, 300]])
    tracker.update("f002.jpg", [])
    tracker.update("f003.jpg", [])
    tracker.update("f004.jpg", [[105, 105, 205, 305]])
    tracker.flush()
    tracks = tracker.all_confirmed_tracks()
    assert len(tracks) == 1
    assert tracks[0].hits >= 2


def test_tracker_flush_returns_remaining():
    tracker = IoUTracker(min_hits=1)
    tracker.update("f001.jpg", [[100, 100, 200, 300]])
    done = tracker.flush()
    assert len(done) == 1


# ─── Direction classification ─────────────────────────────────────────────────

def _make_track(positions: list[tuple]) -> Track:
    return Track(
        track_id=0, bbox=[0, 0, 1, 1],
        hits=len(positions),
        positions=list(positions),
        frame_ids=[f"f{i:03d}.jpg" for i in range(len(positions))],
    )


def test_direction_home():
    positions = [(0.05, 0.5), (0.2, 0.5), (0.5, 0.5), (0.8, 0.5), (0.9, 0.5)]
    d = classify_direction(_make_track(positions), DEFAULT_DOOR_ZONE_X, DEFAULT_ELEVATOR_ZONE_X)
    assert d == DIRECTION_HOME


def test_direction_away():
    positions = [(0.9, 0.5), (0.8, 0.5), (0.5, 0.5), (0.2, 0.5), (0.05, 0.5)]
    d = classify_direction(_make_track(positions), DEFAULT_DOOR_ZONE_X, DEFAULT_ELEVATOR_ZONE_X)
    assert d == DIRECTION_AWAY


def test_direction_loiter_then_home():
    # Ждёт посередине, потом входит в лифт — start у двери, end у лифта → home
    positions = [(0.1, 0.5)] * 3 + [(0.5, 0.5)] * 5 + [(0.85, 0.5)] * 3
    d = classify_direction(_make_track(positions), DEFAULT_DOOR_ZONE_X, DEFAULT_ELEVATOR_ZONE_X)
    assert d == DIRECTION_HOME


def test_direction_too_short():
    d = classify_direction(
        _make_track([(0.1, 0.5), (0.85, 0.5)]),
        DEFAULT_DOOR_ZONE_X, DEFAULT_ELEVATOR_ZONE_X, min_positions=3,
    )
    assert d == DIRECTION_UNKNOWN


def test_direction_middle_only():
    positions = [(0.45 + i * 0.01, 0.5) for i in range(6)]
    d = classify_direction(_make_track(positions), DEFAULT_DOOR_ZONE_X, DEFAULT_ELEVATOR_ZONE_X)
    assert d == DIRECTION_UNKNOWN


# ─── Data loading ─────────────────────────────────────────────────────────────

def _write_detections(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_ts", "run_name", "cam", "filename", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_load_detections_valid(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_detections(run_dir / "detections.csv", [
        ["ts1", "r", "cam1", "f001.jpg", "100", "100", "200", "300", "0.9"],
        ["ts2", "r", "cam1", "f002.jpg", "105", "105", "205", "305", "0.85"],
    ])
    result = load_detections(run_dir)
    assert "cam1" in result
    assert len(result["cam1"]) == 2
    assert result["cam1"][0]["bbox"] == [100, 100, 200, 300]


def test_load_detections_missing(tmp_path: Path):
    assert load_detections(tmp_path / "no_such") == {}


def test_load_detections_sorted(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_detections(run_dir / "detections.csv", [
        ["ts", "r", "cam1", "f003.jpg", "10", "10", "20", "20", "0.8"],
        ["ts", "r", "cam1", "f001.jpg", "10", "10", "20", "20", "0.8"],
        ["ts", "r", "cam1", "f002.jpg", "10", "10", "20", "20", "0.8"],
    ])
    result = load_detections(run_dir)
    frames = [d["frame_id"] for d in result["cam1"]]
    assert frames == sorted(frames)


# ─── Integration ─────────────────────────────────────────────────────────────

def test_run_tracking_home(tmp_path: Path):
    """Один человек идёт слева направо (дверь→лифт) → home.

    Шаг 40px при ширине бокса 80px → IoU ≈ 0.33 > порога 0.25 → треки связываются.
    Зоны заданы в пикселях: дверь x<300, лифт x>700 (центр бокса).
    Кадры: x = 50, 90, ..., 730 (18 кадров).
    """
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    rows = []
    for i, x in enumerate(range(50, 750, 40)):   # 18 кадров, шаг 40px
        rows.append(["ts", "r", "cam1", f"f{i:03d}.jpg",
                     str(x), "100", str(x + 80), "300", "0.9"])
    _write_detections(run_dir / "detections.csv", rows)

    results = run_tracking(
        run_dir,
        Path("no_config.yaml"),
        tmp_path / "out",
        door_x=(0.0, 300.0),      # центр бокса < 300 → у двери
        elevator_x=(700.0, 900.0), # центр бокса > 700 → у лифта
    )

    assert len(results) >= 1
    assert any(r["direction"] == DIRECTION_HOME for r in results)


def test_run_tracking_saves_csv(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    rows = [["ts", "r", "cam1", f"f{i:03d}.jpg", str(i * 10), "100",
             str(i * 10 + 80), "300", "0.9"] for i in range(5)]
    _write_detections(run_dir / "detections.csv", rows)

    out_dir = tmp_path / "out"
    run_tracking(run_dir, Path("no_config.yaml"), out_dir,
                 door_x=(0.0, 50.0), elevator_x=(350.0, 500.0))

    assert (out_dir / "directions.csv").is_file()
    assert (out_dir / "directions.json").is_file()
