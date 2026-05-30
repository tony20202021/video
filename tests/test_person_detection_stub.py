"""
Stub-тесты для логики оценки детекции.
Не требуют модели, камеры или размеченных данных.
Патчат detect_people заглушкой с контролируемым поведением.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Минимальные пороги из основного теста
MIN_PRECISION = 0.5
MIN_RECALL = 0.5


# ─── Фикстура: синтетический датасет во временной директории ─────────────────

@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    Создаёт датасет из синтетических изображений:
    - person/: 4 кадра (шумовые — заглушка будет возвращать детекцию)
    - no_person/: 4 чёрных кадра (заглушка — пустой список)
    """
    person_dir = tmp_path / "person"
    no_person_dir = tmp_path / "no_person"
    person_dir.mkdir()
    no_person_dir.mkdir()

    for i in range(4):
        # Шумовые кадры — будем считать их "с человеком" через заглушку
        frame = np.random.randint(50, 200, (360, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(person_dir / f"person_{i}.jpg"), frame)

    for i in range(4):
        # Чёрные кадры — заглушка вернёт []
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(no_person_dir / f"no_person_{i}.jpg"), frame)

    return tmp_path


def _stub_detect(sess, frame, **kwargs):
    """Заглушка: детектирует человека если кадр шумный (std > 10)."""
    if frame.std() > 10:
        return [(50, 50, 200, 350, 0.85)]
    return []


# ─── Тесты метрик через заглушку ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def _model_stub():
    """Фиктивный объект сессии — передаётся в run_eval вместо реальной."""
    return object()


def _run_eval_with_stub(data_dir: Path, stub_detect) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_person_detection",
        REPO_ROOT / "tests" / "test_person_detection.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with patch("common.utils.person_detector.detect_people", side_effect=stub_detect):
        with patch("common.utils.person_detector.load_model", return_value=object()):
            return mod.run_eval(
                model_path=REPO_ROOT / "models" / "yolov8n.onnx",
                data_dir=data_dir,
                conf=0.35,
            )


def test_precision_stub(synthetic_dataset):
    r = _run_eval_with_stub(synthetic_dataset, _stub_detect)
    # заглушка: person → детекция, no_person → нет → precision должна быть 1.0
    assert r["tp"] == 4
    assert r["fp"] == 0
    assert r["precision"] == pytest.approx(1.0)
    assert r["precision"] >= MIN_PRECISION


def test_recall_stub(synthetic_dataset):
    r = _run_eval_with_stub(synthetic_dataset, _stub_detect)
    assert r["tp"] == 4
    assert r["fn"] == 0
    assert r["recall"] == pytest.approx(1.0)
    assert r["recall"] >= MIN_RECALL


def test_fp_rate_stub(synthetic_dataset):
    r = _run_eval_with_stub(synthetic_dataset, _stub_detect)
    fp_rate = r["fp"] / r["n_no_person"]
    assert fp_rate == 0.0
    assert fp_rate <= 0.20


def test_fp_rate_stub_all_false_positives(synthetic_dataset):
    """Проверяем что при всегда-True заглушке fp_rate = 1.0 и тест падает."""
    def always_detect(sess, frame, **kwargs):
        return [(10, 10, 100, 200, 0.9)]

    r = _run_eval_with_stub(synthetic_dataset, always_detect)
    fp_rate = r["fp"] / r["n_no_person"]
    assert fp_rate == 1.0  # fp_rate > 0.20 → реальный тест должен бы упасть


def test_metrics_calculation(synthetic_dataset):
    """Проверяем корректность формул precision/recall/f1."""
    r = _run_eval_with_stub(synthetic_dataset, _stub_detect)
    tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]

    expected_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    expected_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if (expected_precision + expected_recall) > 0 else 0.0
    )

    assert r["precision"] == pytest.approx(expected_precision)
    assert r["recall"] == pytest.approx(expected_recall)
    assert r["f1"] == pytest.approx(expected_f1)
