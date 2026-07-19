"""Тесты multilabel: формат labels.json (старый/новый), мультихот, пороги."""
from __future__ import annotations

import json

from common.utils import multilabel as ml

ALL = ["1_resident", "2_delivery", "3_utilities", "4_guest"]


def test_normalize_label():
    assert ml.normalize_label("1_resident") == ["1_resident"]         # старый: строка
    assert ml.normalize_label(["1_resident", "2_delivery"]) == ["1_resident", "2_delivery"]
    assert ml.normalize_label("") == []
    assert ml.normalize_label(None) == []
    assert ml.normalize_label([]) == []


def test_load_labels_old_v1(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"version": 1, "labels": {"a.jpg": "1_resident", "b.jpg": "2_delivery"}}),
                 encoding="utf-8")
    got = ml.load_labels(p)
    assert got == {"a.jpg": ["1_resident"], "b.jpg": ["2_delivery"]}   # строка → [строка]


def test_load_labels_new_v2(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"version": 2, "labels": {"a.jpg": ["1_resident", "2_delivery"]}}),
                 encoding="utf-8")
    assert ml.load_labels(p) == {"a.jpg": ["1_resident", "2_delivery"]}


def test_load_labels_list_form(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps([{"image": "a.jpg", "class": "1_resident"},
                             {"image": "b.jpg", "classes": ["2_delivery", "4_guest"]}]),
                 encoding="utf-8")
    assert ml.load_labels(p) == {"a.jpg": ["1_resident"], "b.jpg": ["2_delivery", "4_guest"]}


def test_load_labels_missing(tmp_path):
    assert ml.load_labels(tmp_path / "нет.json") == {}


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "labels.json"
    ml.save_labels(p, {"a.jpg": ["2_delivery", "1_resident", "1_resident"], "b.jpg": ""})
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["version"] == 2
    assert raw["labels"]["a.jpg"] == ["1_resident", "2_delivery"]      # отсортировано, без дублей
    assert raw["labels"]["b.jpg"] == []
    assert ml.load_labels(p)["a.jpg"] == ["1_resident", "2_delivery"]


def test_multihot():
    assert ml.to_multihot(["1_resident", "4_guest"], ALL) == [1, 0, 0, 1]
    assert ml.from_multihot([1, 0, 0, 1], ALL) == ["1_resident", "4_guest"]
    assert ml.to_multihot([], ALL) == [0, 0, 0, 0]


def test_classes_from_probs_float_thr():
    probs = {"1_resident": 0.9, "2_delivery": 0.6, "3_utilities": 0.2, "4_guest": 0.55}
    # порог 0.5 → 3 класса, отсортированы по убыванию вероятности
    assert ml.classes_from_probs(probs, 0.5) == ["1_resident", "2_delivery", "4_guest"]
    assert ml.classes_from_probs(probs, 0.95) == []                   # никто не выше


def test_classes_from_probs_per_class_thr():
    probs = {"1_resident": 0.55, "2_delivery": 0.55}
    thr = {"1_resident": 0.5, "2_delivery": 0.6}     # у delivery порог выше
    assert ml.classes_from_probs(probs, thr) == ["1_resident"]
