"""
Smoke-тесты: проверяют что модель загружается и API не падает.
Не требуют размеченного датасета — работают всегда при наличии yolov8n.onnx.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "yolov8n.onnx"
OUTPUT_DIR = REPO_ROOT / ".output"


@pytest.fixture(scope="module")
def sess():
    if not DEFAULT_MODEL.is_file():
        pytest.skip(f"Модель не найдена: {DEFAULT_MODEL} — запустите python scripts/setup_models.py")
    from common.utils.person_detector import load_model
    s = load_model(DEFAULT_MODEL)
    assert s is not None
    return s


def test_model_loads(sess):
    """Модель загружается и имеет корректный вход."""
    inp = sess.get_inputs()[0]
    assert inp.shape[1] == 3        # RGB
    assert inp.shape[2] == 640      # height
    assert inp.shape[3] == 640      # width


def test_black_frame_no_detections(sess):
    """Чёрный кадр → никаких детекций."""
    from common.utils.person_detector import detect_people
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_people(sess, black)
    assert isinstance(result, list)
    assert result == []


def test_output_format(sess):
    """Формат выхода: список 5-tuple (x1,y1,x2,y2,conf) с float confidence."""
    from common.utils.person_detector import detect_people
    frame = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
    result = detect_people(sess, frame)
    assert isinstance(result, list)
    for item in result:
        x1, y1, x2, y2, conf = item
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
        assert x1 <= x2 and y1 <= y2


def test_draw_boxes_empty(sess):
    """draw_boxes не падает на пустом списке детекций."""
    from common.utils.person_detector import draw_boxes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_boxes(frame, [])
    assert out.shape == frame.shape


def test_draw_boxes_with_detection(sess):
    """draw_boxes не падает при наличии bbox."""
    from common.utils.person_detector import draw_boxes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [(50, 50, 200, 400, 0.91)]
    out = draw_boxes(frame, detections)
    assert out.shape == frame.shape


def test_on_baseline_frame(sess):
    """Запускаем детекцию на реальном baseline-кадре из .output/ (если есть)."""
    import cv2
    from common.utils.person_detector import detect_people

    baselines = sorted(OUTPUT_DIR.rglob("*_baseline.jpg"))
    if not baselines:
        pytest.skip("Нет baseline-кадров в .output/ — запустите 4_motion_watch.py или 5_motion_people.py")

    frame = cv2.imread(str(baselines[0]))
    assert frame is not None, f"Не удалось прочитать {baselines[0]}"
    result = detect_people(sess, frame)
    assert isinstance(result, list)
    print(f"\nBaseline кадр: {baselines[0].name}  детекций: {len(result)}")
