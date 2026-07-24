"""Тест 4_identify_residents: выбор кропов по СГЛАЖЕННЫМ классам из сайдкара smoothed/,
jpg ищется в неизменяемом images/ по имени (+ fallback на физ. раскладку без сайдкара)."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_identify():
    p = REPO_ROOT / "scripts" / "pipeline" / "4_identify_residents.py"
    spec = importlib.util.spec_from_file_location("identify4", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make(tmp_path):
    # images/<date>/ — кропы разложены МОДЕЛЬЮ (single/<model_class>), сглаживание их НЕ трогало
    dd = tmp_path / "images" / "20260720"
    model_dirs = {
        "cam_01_9_d_20260720_085000_100000_msk.jpg": "single/1_resident",
        "cam_01_9_d_20260720_085001_100000_msk.jpg": "single/2_delivery",   # модель: доставка
        "cam_01_9_d_20260720_085002_100000_msk.jpg": "uncertain",            # модель: uncertain
        "cam_01_9_d_20260720_085003_100000_msk.jpg": "single/4_guest",
    }
    for name, sub in model_dirs.items():
        (dd / sub).mkdir(parents=True, exist_ok=True)
        (dd / sub / name).write_bytes(b"x")
    # сайдкар smoothed/<date>/classifications_smoothed.csv — сглаженные классы
    sm = tmp_path / "smoothed" / "20260720"
    sm.mkdir(parents=True, exist_ok=True)
    smoothed = {
        "cam_01_9_d_20260720_085000_100000_msk.jpg": "1_resident",   # → identify
        "cam_01_9_d_20260720_085001_100000_msk.jpg": "1_resident",   # доставка сглажена в резидента → identify
        "cam_01_9_d_20260720_085002_100000_msk.jpg": "1_resident",   # uncertain спасён → identify
        "cam_01_9_d_20260720_085003_100000_msk.jpg": "2_delivery",   # гость сглажен в доставку → НЕ identify
    }
    with open(sm / "classifications_smoothed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "cam", "model_class", "smoothed_class", "changed"])
        w.writeheader()
        for name, scl in smoothed.items():
            w.writerow({"crop": name, "cam": "cam_01_9_d", "model_class": "?",
                        "smoothed_class": scl, "changed": "True"})
    return dd


def test_identify_reads_smoothed_finds_in_images(tmp_path):
    m = _load_identify()
    dd = _make(tmp_path)
    crops = m._v4_identify_crops(dd)
    names = sorted(p.name for p in crops)
    # берутся кропы со СГЛАЖЕННЫМ resident/guest, независимо от папки модели в images/
    assert names == [
        "cam_01_9_d_20260720_085000_100000_msk.jpg",
        "cam_01_9_d_20260720_085001_100000_msk.jpg",   # была доставка, сглажена в резидента
        "cam_01_9_d_20260720_085002_100000_msk.jpg",   # был uncertain, спасён
    ]
    # jpg найдены в РЕАЛЬНЫХ (модельных) местах images/, кропы НЕ переложены
    paths = {p.name: p for p in crops}
    assert paths["cam_01_9_d_20260720_085001_100000_msk.jpg"].parent.name == "2_delivery"
    assert paths["cam_01_9_d_20260720_085002_100000_msk.jpg"].parent.name == "uncertain"
    # гость→доставка НЕ попал в identify
    assert "cam_01_9_d_20260720_085003_100000_msk.jpg" not in paths


def test_identify_fallback_no_smoothed(tmp_path):
    # нет сайдкара smoothed/ → старая физическая раскладка single/{1_resident,4_guest}
    m = _load_identify()
    dd = tmp_path / "images" / "20260720"
    (dd / "single/1_resident").mkdir(parents=True)
    (dd / "single/1_resident/a.jpg").write_bytes(b"x")
    (dd / "single/2_delivery").mkdir(parents=True)
    (dd / "single/2_delivery/b.jpg").write_bytes(b"x")
    crops = m._v4_identify_crops(dd)
    assert [p.name for p in crops] == ["a.jpg"]        # только resident (fallback по папкам)
