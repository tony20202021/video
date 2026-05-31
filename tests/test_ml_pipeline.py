"""Тесты ML-пайплайна: классификатор, идентификатор, pipeline.

Тесты работают без обученных моделей — проверяют интерфейс и дефолтные ответы.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml.classify import GroupClassifier, CLASSES, _preprocess as classify_preprocess
from ml.identify import PersonIdentifier, cosine_similarity, _preprocess as identify_preprocess
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
        # Чёрный кадр: нормализованные значения должны быть отрицательными (сдвиг от mean)
        bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        blob = classify_preprocess(bgr)
        assert (blob < 0).any()

    def test_classes_list(self):
        assert "resident" in CLASSES
        assert "courier" in CLASSES
        assert len(CLASSES) == 5


# ─── PersonIdentifier ─────────────────────────────────────────────────────────

class TestPersonIdentifier:
    def test_no_model_embed_returns_none(self):
        ident = PersonIdentifier()
        assert not ident.ready
        result = ident.embed(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is None

    def test_identify_without_model(self):
        ident = PersonIdentifier()
        pid, sim = ident.identify(np.zeros((100, 100, 3), dtype=np.uint8))
        assert pid is None
        assert sim == 0.0

    def test_add_and_identify_with_mock(self):
        ident = PersonIdentifier()
        # Добавляем эмбеддинги напрямую (без модели)
        emb_alice = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_bob   = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        ident.add_embedding("alice", emb_alice)
        ident.add_embedding("bob", emb_bob)

        # Ищем ближайшего к alice вручную
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        best_id, best_sim = None, -1.0
        for pid, emb in ident._embeddings.items():
            s = cosine_similarity(query, emb)
            if s > best_sim:
                best_id, best_sim = pid, s
        assert best_id == "alice"
        assert best_sim > 0.9

    def test_load_embeddings_from_dict(self):
        ident = PersonIdentifier()
        data = {
            "p_001": [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
            "p_002": [[0.0, 1.0, 0.0]],
        }
        ident.load_embeddings(data)
        assert ident.person_count == 2
        assert "p_001" in ident._embeddings
        assert "p_002" in ident._embeddings

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_preprocess_shape(self):
        bgr = np.random.randint(0, 255, (160, 120, 3), dtype=np.uint8)
        blob = identify_preprocess(bgr)
        assert blob.shape == (1, 3, 112, 112)
        assert blob.dtype == np.float32

    def test_preprocess_range(self):
        bgr = np.zeros((112, 112, 3), dtype=np.uint8)
        blob = identify_preprocess(bgr)
        # Нулевой пиксель → (0 - 127.5) / 128 ≈ -0.996
        assert abs(blob[0, 0, 0, 0] - (-127.5 / 128.0)) < 1e-4


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
