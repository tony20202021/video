"""Тест 4b_smooth_identity: то же ядро (smooth_core), но классы-жители из p_* колонок
identifications.csv; images/ не мутируется, unknown_resident спасается в жителя внутри визита."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PERSONS = ["141_resident_man_1", "141_resident_woman_1"]


def _load4b():
    p = REPO_ROOT / "scripts" / "pipeline" / "4b_smooth_identity.py"
    spec = importlib.util.spec_from_file_location("smooth4b", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make(tmp_path):
    # residents/<ver>/inference/images/<date>/<person|unknown_resident>/crop.jpg  → сайдкар smoothed/<date>/
    dd = tmp_path / "images" / "20260720"
    crops = [
        ("cam_01_9_d_20260720_085000_100000_msk.jpg", "141_resident_man_1", [0.9, 0.1]),
        ("cam_01_9_d_20260720_085001_100000_msk.jpg", "unknown_resident",   [0.55, 0.45]),  # low conf
        ("cam_01_9_d_20260720_085002_100000_msk.jpg", "141_resident_man_1", [0.9, 0.1]),
    ]
    for name, folder, _ in crops:
        (dd / folder).mkdir(parents=True, exist_ok=True)
        (dd / folder / name).write_bytes(b"x")
    fields = ["ts_epoch", "cam", "crop", "person_id", "id_conf"] + [f"p_{p}" for p in PERSONS] + ["out_class"]
    with open(dd / "identifications.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, folder, pr in crops:
            row = {k: "" for k in fields}
            row.update(crop=name, cam="cam_01_9_d",
                       person_id=("" if folder == "unknown_resident" else folder))
            for p, v in zip(PERSONS, pr):
                row[f"p_{p}"] = v
            w.writerow(row)
    return dd


def test_identity_smooths_and_rescues_unknown(tmp_path):
    m = _load4b()
    dd = _make(tmp_path)
    sm = tmp_path / "smoothed" / "20260720"
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, prob_window_sec=3.0)
    assert st["eligible"] == 3          # классы жителей выведены из p_* колонок

    # images/ НЕ мутируется: unknown-кроп остаётся в unknown_resident/
    assert (dd / "unknown_resident/cam_01_9_d_20260720_085001_100000_msk.jpg").is_file()
    assert not (dd / "meta").exists()

    # сайдкар: unknown спасён в жителя (окно соседей-man_1)
    byname = {r["crop"]: r for r in
              csv.DictReader(open(sm / "classifications_smoothed.csv", encoding="utf-8"))}
    assert byname["cam_01_9_d_20260720_085001_100000_msk.jpg"]["smoothed_class"] == "141_resident_man_1"
    assert st["rescued_uncertain"] == 1   # cur=unknown_resident посчитан как спасённый


def test_identity_writes_csv_smoothed_class_not_labels(tmp_path):
    # то же ядро → identity пишет smoothed_class (person) в CSV + p_<person>, но labels.json НЕ пишет
    import csv as _csv
    m = _load4b()
    dd = _make(tmp_path)
    sm = tmp_path / "smoothed" / "20260720"
    m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, prob_window_sec=3.0)

    assert not (sm / "labels.json").exists()                 # labels.json — файл ручной разметки, не сглаживателя

    byname = {r["crop"]: r for r in
              _csv.DictReader(open(sm / "classifications_smoothed.csv", encoding="utf-8"))}
    # unknown-кроп сглажен в жителя
    assert byname["cam_01_9_d_20260720_085001_100000_msk.jpg"]["smoothed_class"] == "141_resident_man_1"
    header = next(_csv.reader(open(sm / "classifications_smoothed.csv", encoding="utf-8")))
    assert header[3] == "smoothed_class"                     # col4 не сдвинулся (identify.sh $4)
    assert "p_141_resident_man_1" in header and "rel" in header
