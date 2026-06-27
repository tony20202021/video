"""
Интеграционные тесты: полный пайплайн на реальных тестовых данных.

Требует: tests/data/ (заполнить через scripts/utils/collect_test_frames.py)
         models/yolov8n.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / ".data"
MODEL_YOLO = REPO_ROOT / ".models" / "yolov8n.onnx"

_NO_PERSON_DIR = TEST_DATA / "no_person"
_WITH_PERSON_DIR = TEST_DATA / "with_person"
_RAW_PAIRS_DIR = TEST_DATA / "raw_pairs"
_ARTIFACTS_DIR = TEST_DATA / "artifacts"

_HAS_TEST_DATA = _NO_PERSON_DIR.exists() and any(_NO_PERSON_DIR.iterdir())
_HAS_YOLO = MODEL_YOLO.is_file()
_HAS_PERSON_FRAMES = _WITH_PERSON_DIR.exists() and any(_WITH_PERSON_DIR.iterdir())


# ─── OSD время на реальных кадрах ────────────────────────────────────────────

class TestOsdTimeIntegration:
    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_osd_time_on_no_person_frames(self):
        """Извлечение OSD времени не падает на обычных кадрах."""
        from common.utils.osd_time import extract_osd_time
        frames = sorted(_NO_PERSON_DIR.glob("*.jpg"))[:5]
        for f in frames:
            img = cv2.imread(str(f))
            assert img is not None
            result = extract_osd_time(img)
            # Результат None или корректный datetime — не должно быть исключений
            if result is not None:
                from datetime import datetime
                assert isinstance(result, datetime)
                assert result.year >= 2020

    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_osd_time_on_artifact_frames_no_crash(self):
        """На битых кадрах OSD не должен падать."""
        from common.utils.osd_time import extract_osd_time
        artifacts = sorted(_ARTIFACTS_DIR.glob("*.jpg"))[:5]
        for f in artifacts:
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = extract_osd_time(img)
            assert result is None or hasattr(result, "year")


# ─── frame_decode_plausible на реальных кадрах ───────────────────────────────

class TestFramePlausibilityIntegration:
    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_normal_frames_are_plausible(self):
        """Нормальные кадры должны проходить фильтр."""
        from common.utils.motion_utils import frame_decode_plausible
        frames = sorted(_NO_PERSON_DIR.glob("*.jpg"))[:10]
        plausible_count = 0
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            if frame_decode_plausible(img, min_laplacian_var=12.0, min_gray_std=2.5):
                plausible_count += 1
        # Большинство нормальных кадров должны быть plausible
        assert plausible_count >= len(frames) * 0.7, (
            f"Слишком мало plausible кадров: {plausible_count}/{len(frames)}"
        )

    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_artifact_frames_no_crash(self):
        """frame_decode_plausible не падает на маленьких/нетипичных кадрах."""
        from common.utils.motion_utils import frame_decode_plausible
        artifacts = sorted(_ARTIFACTS_DIR.glob("*.jpg"))[:10]
        if not artifacts:
            pytest.skip("Нет файлов в artifacts/")
        for f in artifacts:
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = frame_decode_plausible(img, min_laplacian_var=12.0, min_gray_std=2.5)
            assert isinstance(result, bool)


# ─── YOLO на реальных кадрах ─────────────────────────────────────────────────

class TestYoloIntegration:
    @pytest.mark.skipif(not _HAS_YOLO, reason="models/yolov8n.onnx не найден")
    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_yolo_no_person_frames(self):
        """На кадрах без людей YOLO не должен давать много ложных срабатываний."""
        import onnxruntime as ort
        sess = ort.InferenceSession(str(MODEL_YOLO), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        CONF = 0.35
        YOLO_INPUT = 640

        frames = sorted(_NO_PERSON_DIR.glob("*.jpg"))[:10]
        false_positive_frames = 0
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            blob = cv2.resize(img, (YOLO_INPUT, YOLO_INPUT))[:, :, ::-1].astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis]
            output = sess.run(None, {input_name: blob})[0]
            preds = output[0].T
            person_scores = preds[:, 4]
            if (person_scores >= CONF).any():
                false_positive_frames += 1

        # Допускаем не более 40% ложных срабатываний (камера смотрит на пустой коридор)
        assert false_positive_frames <= len(frames) * 0.4, (
            f"Много FP: {false_positive_frames}/{len(frames)}"
        )

    @pytest.mark.skipif(not _HAS_YOLO, reason="models/yolov8n.onnx не найден")
    @pytest.mark.skipif(not _HAS_PERSON_FRAMES, reason="tests/data/with_person/ пуст")
    def test_yolo_detects_persons(self):
        """На кадрах с людьми YOLO должен что-то обнаруживать."""
        import onnxruntime as ort
        sess = ort.InferenceSession(str(MODEL_YOLO), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        CONF = 0.25  # чуть ниже чтобы тест был устойчивее
        YOLO_INPUT = 640

        frames = sorted(_WITH_PERSON_DIR.glob("*.jpg"))
        detected = 0
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            blob = cv2.resize(img, (YOLO_INPUT, YOLO_INPUT))[:, :, ::-1].astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis]
            output = sess.run(None, {input_name: blob})[0]
            preds = output[0].T
            if (preds[:, 4] >= CONF).any():
                detected += 1

        assert detected >= 1, f"YOLO не обнаружил людей ни в одном из {len(frames)} кадров"


# ─── ML пайплайн ─────────────────────────────────────────────────────────────

class TestMLPipelineIntegration:
    @pytest.mark.skipif(not _HAS_TEST_DATA, reason="tests/data/ не заполнен")
    def test_pipeline_no_model_handles_frames(self):
        """Pipeline без моделей обрабатывает кадры без ошибок."""
        from ml.pipeline import MLPipeline, MLConfig
        pipeline = MLPipeline(MLConfig())

        frames = sorted(_NO_PERSON_DIR.glob("*.jpg"))[:3]
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            # Симулируем одну детекцию в центре
            h, w = img.shape[:2]
            fake_dets = [(w // 4, h // 4, 3 * w // 4, 3 * h // 4, 0.8)]
            results = pipeline.run(img, fake_dets)
            assert len(results) == 1
            assert results[0].group_class == "unknown"
            assert results[0].person_id is None
