"""Тесты multi-label конвейера Модели 1 (группы):
  - classify.py: sigmoid/softmax, model-aware вывод, чтение манифеста (multi_label + пороги);
  - 3_train_groups: метрики (AP, подбор порогов, per-class P/R/F1, mAP);
  - 4_identify_residents: маршрутизация входа из v4-раскладки (single/ + multi/ по labels.json);
  - 2_label_ui: разбор датасета v4/legacy, перекладка по набору классов, чтение классов.

Работают без обученных моделей и реальных кропов.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRAIN = _load(REPO_ROOT / "scripts" / "train" / "3_train_groups.py", "t3_train")
IDENT = _load(REPO_ROOT / "scripts" / "pipeline" / "4_identify_residents.py", "p4_ident")
LABELUI = _load(REPO_ROOT / "scripts" / "train" / "2_label_ui.py", "p2_labelui")
BUILD4 = _load(REPO_ROOT / "scripts" / "train" / "build_groups_v4.py", "build_v4")

CLASSES = list(TRAIN.CLASSES)


# ─── classify.py: model-aware вывод + манифест ────────────────────────────────

def test_sigmoid_softmax():
    from ml.classify import _sigmoid, _softmax
    x = np.array([0.0, 0.0])
    assert np.allclose(_sigmoid(x), [0.5, 0.5])
    assert abs(_softmax(x).sum() - 1.0) < 1e-9        # softmax нормируется в 1
    # sigmoid независим по классам — сумма не обязана быть 1
    s = _sigmoid(np.array([2.0, 2.0, 2.0]))
    assert all(v > 0.5 for v in s)


def test_classifier_not_ready():
    from ml.classify import GroupClassifier
    clf = GroupClassifier()                            # без модели
    assert clf.ready is False
    assert clf.multi_label is False
    assert clf.classify(np.zeros((4, 4, 3), np.uint8)) == ("unknown", 0.0, {})
    assert clf.predict_classes(np.zeros((4, 4, 3), np.uint8)) == ([], {})


def test_manifest_multilabel_roundtrip(tmp_path, monkeypatch):
    """versions.write_classify_manifest пишет multi_label+thresholds; classify их читает."""
    from ml import versions
    from ml.classify import GroupClassifier
    monkeypatch.setattr(versions, "_MODELS_DIR", tmp_path)
    thr = {c: 0.4 for c in CLASSES}
    versions.write_classify_manifest("v4_1", multi_label=True, thresholds=thr)
    # модель не загружаем (нет onnx) — вызываем _load_manifest напрямую
    clf = GroupClassifier()
    clf._load_manifest(tmp_path / "v4_1.onnx")
    assert clf.multi_label is True
    assert clf.thresholds == thr


def test_manifest_single_label_default(tmp_path):
    from ml.classify import GroupClassifier
    (tmp_path / "v1_1.json").write_text(json.dumps({"tag": "v1_1"}), encoding="utf-8")
    clf = GroupClassifier()
    clf._load_manifest(tmp_path / "v1_1.onnx")
    assert clf.multi_label is False           # нет поля → single-label (softmax)
    assert clf.thresholds == {}


# ─── 3_train_groups: метрики multi-label ──────────────────────────────────────

def test_average_precision():
    # идеальное ранжирование позитивов → AP = 1
    assert TRAIN._average_precision(np.array([1, 1, 0, 0.]), np.array([.9, .8, .2, .1])) == 1.0
    # нет позитивов → AP = 0
    assert TRAIN._average_precision(np.array([0, 0, 0.]), np.array([.9, .5, .1])) == 0.0
    # худшее ранжирование → AP < 1
    ap = TRAIN._average_precision(np.array([0, 0, 1, 1.]), np.array([.9, .8, .2, .1]))
    assert 0.0 < ap < 1.0


def test_tune_thresholds():
    C = len(CLASSES)
    n = 20
    yt = np.zeros((n, C)); ys = np.zeros((n, C))
    # класс 0: чётко разделим при пороге ~0.5
    yt[:10, 0] = 1; ys[:10, 0] = 0.9; ys[10:, 0] = 0.1
    thr = TRAIN._tune_thresholds(yt, ys)
    assert 0.1 < thr[CLASSES[0]] <= 0.9
    # класс без позитивов → дефолтный порог
    assert thr[CLASSES[1]] == TRAIN.ml.DEFAULT_THRESHOLD


def test_multilabel_metrics():
    yt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], float)
    ys = np.array([[.9, .1, .1, .1], [.1, .8, .1, .1],
                   [.95, .1, .1, .7], [.1, .1, .1, .1]], float)
    m = TRAIN._multilabel_metrics(yt, ys, {c: 0.5 for c in CLASSES})
    assert m["subset_accuracy"] == 1.0                 # при 0.5 все предсказания точны
    assert m["by_class"]["1_resident"]["f1"] == 1.0
    assert m["by_class"]["1_resident"]["support"] == 2
    assert m["by_class"]["3_utilities"]["support"] == 0
    assert 0.0 <= m["mAP"] <= 1.0


def test_load_dataset_v4_multi(tmp_path):
    """_load_dataset читает v4 (single/+multi/+labels.json) как списки классов."""
    ds = tmp_path / "dataset"
    (ds / "single" / "1_resident").mkdir(parents=True)
    (ds / "multi").mkdir()
    (ds / "single" / "1_resident" / "a.jpg").write_bytes(b"x")
    (ds / "multi" / "b.jpg").write_bytes(b"x")
    (ds / "labels.json").write_text(json.dumps({"version": 2, "labels": {
        "single/1_resident/a.jpg": ["1_resident"],
        "multi/b.jpg": ["1_resident", "2_delivery"],
    }}), encoding="utf-8")
    base, labels = TRAIN._load_dataset(ds)
    by = {lb["image"].split("/")[-1]: lb["classes"] for lb in labels}
    assert by == {"a.jpg": ["1_resident"], "b.jpg": ["1_resident", "2_delivery"]}


def test_load_dataset_legacy_single(tmp_path):
    """Старый folder-based датасет → 1-элементные списки классов."""
    ds = tmp_path / "v1"
    (ds / "1_resident").mkdir(parents=True)
    (ds / "1_resident" / "a.jpg").write_bytes(b"x")
    base, labels = TRAIN._load_dataset(ds)
    assert labels == [{"image": "1_resident/a.jpg", "classes": ["1_resident"]}]


def test_load_dataset_excludes_skip(tmp_path):
    """Кропы, помеченные skip/unknown/new, исключаются из обучения."""
    ds = tmp_path / "dataset"
    (ds / "single" / "1_resident").mkdir(parents=True)
    (ds / "skip").mkdir()
    (ds / "single" / "1_resident" / "a.jpg").write_bytes(b"x")
    (ds / "skip" / "b.jpg").write_bytes(b"x")
    (ds / "labels.json").write_text(json.dumps({"version": 2, "labels": {
        "single/1_resident/a.jpg": ["1_resident"],
        "skip/b.jpg": ["skip"],
    }}), encoding="utf-8")
    base, labels = TRAIN._load_dataset(ds)
    assert {lb["image"].split("/")[-1] for lb in labels} == {"a.jpg"}   # skip исключён


def test_labelui_target_dir_skip(tmp_path):
    ds = tmp_path / "dataset"
    assert LABELUI._v4_target_dir(ds, ["skip"]) == ds / "skip"
    assert LABELUI._v4_target_dir(ds, []) == ds / "skip"
    assert LABELUI._v4_target_dir(ds, ["1_resident"]) == ds / "single" / "1_resident"
    assert LABELUI._v4_target_dir(ds, ["1_resident", "2_delivery"]) == ds / "multi"


def test_build4_parse_and_clean():
    cam, date, sod = BUILD4._parse_name("cam_01_9_d_20260628_143742_559207_msk_diff5.1.jpg")
    assert (cam, date) == ("cam_01_9_d", "20260628")
    assert sod == 14 * 3600 + 37 * 60 + 42
    assert BUILD4._parse_name("мусор.jpg") is None
    assert BUILD4._clean_classes(["1_resident", "uncertain"]) == ["1_resident"]  # псевдо убрано
    assert BUILD4._clean_classes(["skip"]) == []
    assert BUILD4._clean_classes(["4_guest", "1_resident"]) == ["1_resident", "4_guest"]


def test_build4_farthest_point(tmp_path):
    import cv2
    # 4 кропа: c0==c1 (одинаковые), c2/c3 разные → FPS должен взять разнообразные, дубль отбросить
    items = {}
    names = []
    for i, v in enumerate((0, 0, 128, 255)):
        p = tmp_path / f"c{i}.jpg"
        cv2.imwrite(str(p), np.full((48, 48, 3), v, np.uint8))
        names.append(f"c{i}.jpg")
        items[f"c{i}.jpg"] = {"path": p}
    sel = BUILD4._farthest_point(names, items, 3)
    assert len(sel) == 3
    assert "c2.jpg" in sel and "c3.jpg" in sel        # разные — взяты
    assert not ("c0.jpg" in sel and "c1.jpg" in sel)  # оба идентичных — нет
    # k >= размера → все
    assert set(BUILD4._farthest_point(names, items, 9)) == set(names)


def test_labelui_clean_label_set():
    """Настоящий класс и псевдо-метки (skip/unknown/uncertain/new) взаимоисключающи."""
    c = LABELUI._clean_label_set
    assert c(["1_resident", "uncertain"]) == ["1_resident"]      # класс → uncertain убрать
    assert c(["1_resident", "skip"]) == ["1_resident"]
    assert c(["4_guest", "1_resident", "unknown"]) == ["1_resident", "4_guest"]
    assert c(["uncertain"]) == ["uncertain"]                     # только псевдо — оставить
    assert c(["skip"]) == ["skip"]
    assert c([]) == []
    assert "uncertain" in set(LABELUI.EXTRA_DATASET_DIRS)        # uncertain — псевдо-метка


# ─── 4_identify_residents: v4-маршрутизация ───────────────────────────────────

def test_v4_identify_routing(tmp_path):
    """single/1_resident + single/4_guest + multi/ (только с resident/guest в labels.json)."""
    rd = tmp_path
    for p in ["single/1_resident", "single/4_guest", "single/2_delivery", "multi", "uncertain"]:
        (rd / p).mkdir(parents=True)
    (rd / "single/1_resident/r.jpg").write_bytes(b"x")
    (rd / "single/4_guest/g.jpg").write_bytes(b"x")
    (rd / "single/2_delivery/d.jpg").write_bytes(b"x")     # НЕ на идентификацию
    (rd / "multi/m_ok.jpg").write_bytes(b"x")              # содержит resident
    (rd / "multi/m_no.jpg").write_bytes(b"x")              # delivery+utilities
    (rd / "uncertain/u.jpg").write_bytes(b"x")             # НЕ на идентификацию
    (rd / "labels.json").write_text(json.dumps({"version": 2, "labels": {
        "single/1_resident/r.jpg": ["1_resident"],
        "single/4_guest/g.jpg": ["4_guest"],
        "single/2_delivery/d.jpg": ["2_delivery"],
        "multi/m_ok.jpg": ["1_resident", "2_delivery"],
        "multi/m_no.jpg": ["2_delivery", "3_utilities"],
    }}), encoding="utf-8")
    crops, flat = IDENT._find_identify_crops(rd)
    names = sorted(c.name for c, _ in crops)
    assert flat is True
    assert names == ["g.jpg", "m_ok.jpg", "r.jpg"]
    assert IDENT._has_identify_crops(rd) is True


def test_v4_identify_multi_without_labels_conservative(tmp_path):
    """Нет labels.json → берём все multi-кропы (консервативно)."""
    rd = tmp_path
    (rd / "multi").mkdir(parents=True)
    (rd / "multi" / "x.jpg").write_bytes(b"x")
    crops = IDENT._v4_identify_crops(rd)
    assert [c.name for c in crops] == ["x.jpg"]


# ─── 2_label_ui: датасет v4/legacy + перекладка + классы ──────────────────────

def test_labelui_dataset_files_v4(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "single" / "1_resident").mkdir(parents=True)
    (ds / "multi").mkdir()
    (ds / "single" / "1_resident" / "a.jpg").write_bytes(b"x")
    (ds / "multi" / "c.jpg").write_bytes(b"x")
    (ds / "labels.json").write_text(json.dumps({"version": 2, "labels": {
        "single/1_resident/a.jpg": ["1_resident"],
        "multi/c.jpg": ["1_resident", "2_delivery"],
    }}), encoding="utf-8")
    files, layout = LABELUI._dataset_files(ds, {".jpg"})
    assert layout == "v4"
    by = {Path(f["path"]).name: f["classes"] for f in files}
    assert by == {"a.jpg": ["1_resident"], "c.jpg": ["1_resident", "2_delivery"]}


def test_labelui_relocate_and_labels(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "single" / "1_resident").mkdir(parents=True)
    (ds / "single" / "1_resident" / "a.jpg").write_bytes(b"x")
    dmap = {"single/1_resident/a.jpg": ["1_resident"]}
    # добавили класс → переезд в multi/
    dst = LABELUI._v4_relocate(ds, ds / "single/1_resident/a.jpg",
                               ["1_resident", "4_guest"], dmap)
    assert dst == ds / "multi" / "a.jpg"
    assert dmap == {"multi/a.jpg": ["1_resident", "4_guest"]}
    assert dst.exists() and not (ds / "single/1_resident/a.jpg").exists()
    # очистили набор → skip/, ключ удалён
    dst2 = LABELUI._v4_relocate(ds, dst, [], dmap)
    assert dst2 == ds / "skip" / "a.jpg"
    assert dmap == {}


def test_labelui_load_classes(tmp_path):
    ds = tmp_path / "v4" / "dataset"
    (ds / "single" / "1_resident").mkdir(parents=True)
    (ds / "multi").mkdir()
    classes, resolved = LABELUI._load_classes(ds)
    assert classes == CLASSES                          # все группы доступны кнопками
    # legacy
    ds2 = tmp_path / "legacy"
    (ds2 / "1_resident").mkdir(parents=True)
    (ds2 / "skip").mkdir()
    c2, _ = LABELUI._load_classes(ds2)
    assert c2 == ["1_resident"]                        # skip исключён
