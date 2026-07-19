"""Тесты для скриптов 3_classify_groups и 4_identify_residents.

Проверяют логику обнаружения входных данных, классификацию без модели и
парсинг имён файлов. Работают без обученных моделей и без реальных кропов.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "pipeline"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script(name: str):
    """Импортирует модуль из scripts/pipeline/ по имени файла."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m62():
    return _load_script("3_classify_groups")


@pytest.fixture(scope="module")
def m63():
    return _load_script("4_identify_residents")


# ─── 6_2: _has_crops ─────────────────────────────────────────────────────────

class TestHasCrops:
    def test_empty_dir_false(self, m62, tmp_path):
        assert m62._has_crops(tmp_path) is False

    def test_valid_5_2_structure(self, m62, tmp_path):
        (tmp_path / "run_20260628_143650_msk" / "cam_01_d" / "crops").mkdir(parents=True)
        assert m62._has_crops(tmp_path) is True

    def test_no_crops_subdir_false(self, m62, tmp_path):
        (tmp_path / "run_20260628_143650_msk" / "cam_01_d" / "frames").mkdir(parents=True)
        assert m62._has_crops(tmp_path) is False

    def test_non_run_subdirs_ignored(self, m62, tmp_path):
        (tmp_path / "other_dir" / "cam_01_d" / "crops").mkdir(parents=True)
        assert m62._has_crops(tmp_path) is False


# ─── 6_2: _find_crops ────────────────────────────────────────────────────────

class TestFindCrops:
    def _make_crop(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff")  # минимальный JPEG-заголовок

    def test_finds_crops_in_structure(self, m62, tmp_path):
        crop = tmp_path / "run_001" / "cam_d" / "crops" / "img_001.jpg"
        self._make_crop(crop)

        result = m62._find_crops(tmp_path)
        assert "run_001" in result
        assert "cam_d" in result["run_001"]
        assert len(result["run_001"]["cam_d"]) == 1

    def test_ignores_non_run_subdirs(self, m62, tmp_path):
        (tmp_path / "logs" / "cam_d" / "crops" / "img.jpg").parent.mkdir(parents=True)
        result = m62._find_crops(tmp_path)
        assert result == {}

    def test_multiple_cams_multiple_subruns(self, m62, tmp_path):
        for sub in ["run_001", "run_002"]:
            for cam in ["cam_d", "cam_u"]:
                crop = tmp_path / sub / cam / "crops" / "img.jpg"
                self._make_crop(crop)

        result = m62._find_crops(tmp_path)
        assert set(result.keys()) == {"run_001", "run_002"}
        for sub in result.values():
            assert set(sub.keys()) == {"cam_d", "cam_u"}

    def test_empty_crops_dir_not_included(self, m62, tmp_path):
        (tmp_path / "run_001" / "cam_d" / "crops").mkdir(parents=True)
        result = m62._find_crops(tmp_path)
        assert result == {}


# ─── 6_2: _classify ──────────────────────────────────────────────────────────

class _FakeSingle:
    """Single-label модель: predict_classes → [argmax] (порог уверенности применяет _classify)."""
    ready = True
    multi_label = False
    def __init__(self, cls, conf, pm=None):
        self._cls, self._pm = cls, pm or {cls: conf}
    def predict_classes(self, bgr):
        return [self._cls], self._pm


class _FakeMulti:
    """Multi-label модель: predict_classes → НАБОР классов (пороги уже применены)."""
    ready = True
    multi_label = True
    def __init__(self, classes, pm):
        self._classes, self._pm = classes, pm
    def predict_classes(self, bgr):
        return self._classes, self._pm


class TestClassify:
    def test_no_classifier_returns_unknown(self, m62):
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(None, crop, classify_conf=0.65)
        assert res["classes"] == []
        assert res["group"] == "unknown"
        assert res["group_conf"] == 0.0
        assert res["prob_map"] == {}
        # модель не готова → unknown/
        assert m62._placement(res["classes"], False) == ("unknown", __import__("pathlib").Path("unknown"))

    def test_single_low_conf_goes_to_uncertain(self, m62):
        """single-label: conf ниже порога → пустой набор → uncertain/."""
        clf = _FakeSingle("1_resident", 0.4, {"1_resident": 0.4, "2_delivery": 0.3})
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(clf, crop, classify_conf=0.65)
        assert res["classes"] == []
        assert res["group"] == "1_resident"
        assert m62._placement(res["classes"], True)[0] == "uncertain"

    def test_single_high_conf_uses_group_class(self, m62):
        clf = _FakeSingle("2_delivery", 0.82, {"2_delivery": 0.82, "1_resident": 0.12})
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(clf, crop, classify_conf=0.65)
        assert res["classes"] == ["2_delivery"]
        assert abs(res["conf_2nd"] - 0.12) < 1e-6
        # 1 класс → single/<class>/
        out_class, rel = m62._placement(res["classes"], True)
        assert out_class == "2_delivery"
        assert rel.as_posix() == "single/2_delivery"

    def test_single_exact_threshold_passes(self, m62):
        clf = _FakeSingle("2_delivery", 0.65, {"2_delivery": 0.65})
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(clf, crop, classify_conf=0.65)
        assert res["classes"] == ["2_delivery"]

    def test_multi_returns_set_to_multi_dir(self, m62):
        """multi-label: несколько классов → multi/, to_identify по 1_resident/4_guest."""
        clf = _FakeMulti(["1_resident", "2_delivery"],
                         {"1_resident": 0.9, "2_delivery": 0.7, "3_utilities": 0.1, "4_guest": 0.2})
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(clf, crop, classify_conf=0.65)
        assert res["classes"] == ["1_resident", "2_delivery"]
        assert m62._placement(res["classes"], True)[0] == "multi"
        row = m62._make_row(res, "multi", mono_s=0, ts_epoch=None,
                            run_name="r", sub_run="s", cam="c", crop="x.jpg")
        assert row["classes"] == "1_resident|2_delivery"
        assert row["n_classes"] == 2
        assert row["to_identify"] is True

    def test_multi_empty_set_is_uncertain(self, m62):
        clf = _FakeMulti([], {"1_resident": 0.1, "2_delivery": 0.2,
                              "3_utilities": 0.1, "4_guest": 0.15})
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        res = m62._classify(clf, crop, classify_conf=0.65)
        assert res["classes"] == []
        assert m62._placement(res["classes"], True)[0] == "uncertain"


# ─── 6_2: _crop_ts_epoch ─────────────────────────────────────────────────────

class TestCropTsEpoch:
    MSK = timezone(timedelta(hours=3))

    def test_parses_valid_filename(self, m62):
        name = "cam_01_9_d_20260628_143742_559207_msk_diff5.1_p1of1_conf0.49.jpg"
        ts = m62._crop_ts_epoch(Path(name))
        assert ts is not None
        dt = datetime.fromtimestamp(ts, tz=self.MSK)
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.day == 28
        assert dt.hour == 14
        assert dt.minute == 37
        assert dt.second == 42

    def test_returns_none_for_unknown_format(self, m62):
        ts = m62._crop_ts_epoch(Path("random_file.jpg"))
        assert ts is None

    def test_returns_none_for_partial_timestamp(self, m62):
        ts = m62._crop_ts_epoch(Path("frame_20260628.jpg"))
        assert ts is None


# ─── 6_3: _has_resident_crops ────────────────────────────────────────────────

class TestHasResidentCrops:
    def test_empty_dir_false(self, m63, tmp_path):
        assert m63._has_resident_crops(tmp_path) is False

    def test_with_resident_dir(self, m63, tmp_path):
        d = tmp_path / "sub" / "cam" / "classified" / "1_resident"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"\xff\xd8\xff")
        assert m63._has_resident_crops(tmp_path) is True

    def test_only_other_class_dir_false(self, m63, tmp_path):
        (tmp_path / "sub" / "cam" / "classified" / "2_delivery").mkdir(parents=True)
        assert m63._has_resident_crops(tmp_path) is False


# ─── 6_3: _find_resident_crops ───────────────────────────────────────────────

class TestFindResidentCrops:
    def _make_crop(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff")

    def test_finds_resident_crops(self, m63, tmp_path):
        crop = tmp_path / "sub" / "cam_d" / "classified" / "resident" / "img.jpg"
        self._make_crop(crop)

        results = m63._find_resident_crops(tmp_path)
        assert len(results) == 1
        found_path, rel_cam = results[0]
        assert found_path.name == "img.jpg"
        assert rel_cam == Path("sub") / "cam_d"

    def test_ignores_non_jpg(self, m63, tmp_path):
        (tmp_path / "sub" / "cam_d" / "classified" / "resident").mkdir(parents=True)
        (tmp_path / "sub" / "cam_d" / "classified" / "resident" / "img.png").write_bytes(b"")
        results = m63._find_resident_crops(tmp_path)
        assert results == []

    def test_multiple_resident_dirs(self, m63, tmp_path):
        for cam in ["cam_d", "cam_u"]:
            crop = tmp_path / "sub1" / cam / "classified" / "resident" / "img.jpg"
            self._make_crop(crop)

        results = m63._find_resident_crops(tmp_path)
        assert len(results) == 2

    def test_ignores_other_class_dirs(self, m63, tmp_path):
        for cls in ["2_delivery", "99_other", "uncertain"]:
            crop = tmp_path / "sub" / "cam" / "classified" / cls / "img.jpg"
            self._make_crop(crop)

        results = m63._find_resident_crops(tmp_path)
        assert results == []

    def test_empty_resident_dir(self, m63, tmp_path):
        (tmp_path / "sub" / "cam" / "classified" / "resident").mkdir(parents=True)
        results = m63._find_resident_crops(tmp_path)
        assert results == []


# ─── 6_3: _identify ──────────────────────────────────────────────────────────

class TestIdentify:
    def test_no_identifier_returns_unknown_resident(self, m63):
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        pid, conf, out_class, probs = m63._identify(None, crop, identify_conf=0.70)
        assert pid is None
        assert conf == 0.0
        assert out_class == "unknown_resident"
        assert probs == {}

    def test_low_conf_returns_unknown_resident(self, m63):
        class _FakeIdentLow:
            ready = True
            def identify_with_probs(self, bgr, threshold):
                return None, 0.5, {}  # ниже порога

        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        pid, conf, out_class, probs = m63._identify(_FakeIdentLow(), crop, identify_conf=0.70)
        assert pid is None
        assert out_class == "unknown_resident"

    def test_high_conf_returns_person_id(self, m63):
        class _FakeIdentHigh:
            ready = True
            def identify_with_probs(self, bgr, threshold):
                return "person_01", 0.91, {"person_01": 0.91}

        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        pid, conf, out_class, probs = m63._identify(_FakeIdentHigh(), crop, identify_conf=0.70)
        assert pid == "person_01"
        assert abs(conf - 0.91) < 1e-6
        assert out_class == "person_01"
