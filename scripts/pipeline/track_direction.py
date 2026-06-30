"""Ядро трекинга и классификации направления движения.

Импортируется тестами и скриптом 5_track_direction.py.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.pipeline.tracker import IoUTracker, Track

DEFAULT_DOOR_ZONE_X     = (0.0, 0.30)
DEFAULT_ELEVATOR_ZONE_X = (0.70, 1.0)

DIRECTION_HOME    = "home"
DIRECTION_AWAY    = "away"
DIRECTION_UNKNOWN = "unknown"


def load_zones(config_path: Path) -> tuple[tuple, tuple]:
    door_x     = DEFAULT_DOOR_ZONE_X
    elevator_x = DEFAULT_ELEVATOR_ZONE_X
    if not config_path.is_file():
        return door_x, elevator_x
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        zones = cfg.get("tracking", {}).get("zones", {})
        if "door_x" in zones:
            door_x = tuple(zones["door_x"])
        if "elevator_x" in zones:
            elevator_x = tuple(zones["elevator_x"])
    except Exception as e:
        import sys
        print(f"  [!] Ошибка чтения зон из {config_path}: {e}", file=sys.stderr)
    return door_x, elevator_x


def load_tracker_params(config_path: Path) -> dict:
    params = {"iou_threshold": 0.25, "max_age": 5, "min_hits": 2}
    if not config_path.is_file():
        return params
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tp = cfg.get("tracking", {}).get("tracker", {})
        params.update({k: tp[k] for k in params if k in tp})
    except Exception:
        pass
    return params


def _in_zone(cx: float, x_range: tuple) -> bool:
    return x_range[0] <= cx <= x_range[1]


def classify_direction(
    track: Track,
    door_x: tuple,
    elevator_x: tuple,
    min_positions: int = 3,
) -> str:
    """Определить направление по истории позиций трека.

    Берём первую и последнюю четверть трека, чтобы нивелировать
    ожидание у лифта в середине маршрута.

    Returns: "home" / "away" / "unknown"
    """
    if len(track.positions) < min_positions:
        return DIRECTION_UNKNOWN

    n = len(track.positions)
    q = max(1, n // 4)
    start_cx = sum(p[0] for p in track.positions[:q]) / q
    end_cx   = sum(p[0] for p in track.positions[-q:]) / q

    if _in_zone(start_cx, door_x) and _in_zone(end_cx, elevator_x):
        return DIRECTION_HOME
    if _in_zone(start_cx, elevator_x) and _in_zone(end_cx, door_x):
        return DIRECTION_AWAY
    return DIRECTION_UNKNOWN


def load_detections(run_dir: Path) -> dict[str, list[dict]]:
    """Загрузить detections.csv.

    Returns {cam: [{"frame_id", "bbox": [x1,y1,x2,y2], "conf"}, ...]}
    """
    det_csv = run_dir / "detections.csv"
    if not det_csv.is_file():
        return {}

    by_cam: dict[str, list[dict]] = defaultdict(list)
    with open(det_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                by_cam[row["cam"]].append({
                    "frame_id": row["filename"],
                    "bbox": [float(row["x1"]), float(row["y1"]),
                             float(row["x2"]), float(row["y2"])],
                    "conf": float(row["conf"]),
                })
            except (KeyError, ValueError):
                continue

    for cam in by_cam:
        by_cam[cam].sort(key=lambda r: r["frame_id"])

    return dict(by_cam)


def run_tracking(
    run_dir: Path,
    config_path: Path,
    out_dir: Path,
    *,
    door_x: tuple | None = None,
    elevator_x: tuple | None = None,
) -> list[dict]:
    """Трекинг по detections.csv, сохранение directions.csv / .json.

    door_x / elevator_x переопределяют config (для тестов).
    """
    import sys

    _door_x, _elev_x = load_zones(config_path)
    if door_x is not None:
        _door_x = door_x
    if elevator_x is not None:
        _elev_x = elevator_x

    tracker_params = load_tracker_params(config_path)
    detections_by_cam = load_detections(run_dir)

    if not detections_by_cam:
        print(f"  [!] detections.csv не найден в {run_dir}", file=sys.stderr)
        return []

    results: list[dict] = []

    for cam, det_list in sorted(detections_by_cam.items()):
        print(f"  {cam}: {len(det_list)} детекций")

        tracker = IoUTracker(**tracker_params)
        by_frame: dict[str, list] = defaultdict(list)
        for det in det_list:
            by_frame[det["frame_id"]].append(det["bbox"])

        for frame_id in sorted(by_frame):
            tracker.update(frame_id, by_frame[frame_id])
        tracker.flush()

        tracks = tracker.all_confirmed_tracks()
        print(f"    треков: {len(tracks)}")

        for track in tracks:
            direction = classify_direction(track, _door_x, _elev_x)
            sc, ec = track.start_center, track.end_center
            results.append({
                "run":       run_dir.name,
                "cam":       cam,
                "track_id":  track.track_id,
                "hits":      track.hits,
                "direction": direction,
                "start_cx":  round(sc[0], 4) if sc else None,
                "end_cx":    round(ec[0], 4) if ec else None,
                "n_frames":  len(track.frame_ids),
            })
            sym = {"home": "→🏠", "away": "🚪→"}.get(direction, "?")
            print(f"    track {track.track_id}: {direction} {sym}  "
                  f"hits={track.hits}  cx: {sc and round(sc[0], 2)} → {ec and round(ec[0], 2)}")

    if results and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "directions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        (out_dir / "directions.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return results
