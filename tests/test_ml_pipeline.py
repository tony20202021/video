"""Тесты ML-пайплайна: классификатор, идентификатор, pipeline.

Тесты работают без обученных моделей — проверяют интерфейс и дефолтные ответы.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml.classify import GroupClassifier, CLASSES, _preprocess as classify_preprocess
from ml.identify import PersonIdentifier, _preprocess as identify_preprocess
from ml.pipeline import MLPipeline, MLConfig, PersonResult, _extract_crop


# ─── GroupClassifier ──────────────────────────────────────────────────────────

class TestGroupClassifier:
    def test_no_model_returns_unknown(self):
        clf = GroupClassifier()
        assert not clf.ready
        cls, conf, probs = clf.classify(np.zeros((100, 60, 3), dtype=np.uint8))
        assert cls == "unknown"
        assert conf == 0.0
        assert probs == {}

    def test_load_nonexistent_returns_false(self):
        clf = GroupClassifier()
        ok = clf.load(Path("/nonexistent/model.onnx"))
        assert ok is False
        assert not clf.ready

    def test_preprocess_shape(self):
        bgr = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        blob = classify_preprocess(bgr)
        assert blob.shape == (1, 3, 224, 224)
        assert blob.dtype == np.float32

    def test_preprocess_normalization(self):
        bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        blob = classify_preprocess(bgr)
        assert (blob < 0).any()

    def test_classes_list(self):
        assert "1_resident" in CLASSES
        assert "2_delivery" in CLASSES
        assert "courier" not in CLASSES
        assert len(CLASSES) == 4


# ─── PersonIdentifier ─────────────────────────────────────────────────────────

class TestPersonIdentifier:
    def test_no_model_not_ready(self):
        ident = PersonIdentifier()
        assert not ident.ready

    def test_person_count_without_model(self):
        ident = PersonIdentifier()
        assert ident.person_count == 0

    def test_identify_without_model_returns_none(self):
        ident = PersonIdentifier()
        pid, conf = ident.identify(np.zeros((100, 100, 3), dtype=np.uint8))
        assert pid is None
        assert conf == 0.0

    def test_identify_with_explicit_classes(self):
        ident = PersonIdentifier(classes=["alice", "bob"])
        # Без модели всё равно (None, 0.0) — сессии нет
        assert not ident.ready
        pid, conf = ident.identify(np.zeros((100, 100, 3), dtype=np.uint8))
        assert pid is None
        assert conf == 0.0

    def test_load_nonexistent_returns_false(self):
        ident = PersonIdentifier()
        ok = ident.load(Path("/nonexistent/model.onnx"))
        assert ok is False
        assert not ident.ready

    def test_preprocess_shape(self):
        bgr = np.random.randint(0, 255, (160, 120, 3), dtype=np.uint8)
        blob = identify_preprocess(bgr)
        assert blob.shape == (1, 3, 224, 224)
        assert blob.dtype == np.float32

    def test_preprocess_normalization(self):
        bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        blob = identify_preprocess(bgr)
        assert (blob < 0).any()


# ─── MLPipeline ───────────────────────────────────────────────────────────────

class TestMLPipeline:
    def _make_pipeline(self) -> MLPipeline:
        return MLPipeline(MLConfig())

    def test_run_empty_detections(self):
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.run(frame, [])
        assert results == []

    def test_run_returns_person_results(self):
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [(10, 10, 100, 200, 0.9), (200, 50, 350, 300, 0.75)]
        results = pipeline.run(frame, detections)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, PersonResult)
            assert r.group_class == "unknown"  # без модели
            assert r.person_id is None

    def test_run_preserves_bbox_and_conf(self):
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = [(50, 60, 150, 250, 0.88)]
        results = pipeline.run(frame, det)
        assert results[0].bbox == (50, 60, 150, 250)
        assert abs(results[0].detect_conf - 0.88) < 1e-6

    def test_classify_crop_no_model(self):
        pipeline = self._make_pipeline()
        crop = np.zeros((64, 48, 3), dtype=np.uint8)
        r = pipeline.classify_crop(crop)
        assert isinstance(r, PersonResult)
        assert r.group_class == "unknown"
        assert r.person_id is None
        assert r.bbox == (0, 0, 0, 0)

    def test_extract_crop_clamps_to_frame(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        crop = _extract_crop(frame, -10, -10, 50, 50, 0.1, 100, 100)
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_from_config(self):
        cfg = {
            "models": {},
            "thresholds": {
                "classification": {"min_confidence": 0.7},
                "identification": {"min_confidence": 0.8},
            },
        }
        pipeline = MLPipeline.from_config(cfg)
        assert pipeline._cfg.classify_threshold == 0.7
        assert pipeline._cfg.identify_threshold == 0.8
