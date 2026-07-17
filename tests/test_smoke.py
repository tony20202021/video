"""
Smoke-тесты обёртки детекции — на StubSession, БЕЗ реальной модели и onnxruntime.

Проверяется контракт load/detect/postprocess/draw на управляемой заглушке:
заглушка отдаёт синтетический YOLOv8-выход, а `detect_people` прогоняет реальные
`_preprocess`/`_postprocess`. Тесты на реальной модели — в test_person_detection.py
(маркер integration).
"""

from __future__ import annotations

import numpy as np
import pytest

from common.utils.person_detector import detect_people, draw_boxes


class _StubInput:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class StubSession:
    """Заглушка onnxruntime.InferenceSession.

    boxes — список (cx, cy, bw, bh, conf) в координатах входа 640×640;
    run() отдаёт синтетический YOLOv8-выход [1, 84, 8400].
    """

    def __init__(self, boxes=()):
        self._boxes = list(boxes)

    def get_inputs(self):
        return [_StubInput("images", [1, 3, 640, 640])]

    def run(self, output_names, feeds):
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        for i, (cx, cy, bw, bh, conf) in enumerate(self._boxes):
            out[0, 0, i] = cx
            out[0, 1, i] = cy
            out[0, 2, i] = bw
            out[0, 3, i] = bh
            out[0, 4, i] = conf  # скор класса person (индекс 4)
        return [out]


@pytest.fixture
def stub_empty():
    return StubSession(boxes=[])


@pytest.fixture
def stub_one():
    return StubSession(boxes=[(320.0, 320.0, 100.0, 200.0, 0.9)])


def test_stub_input_shape(stub_empty):
    """Вход сессии: [1, 3, 640, 640]."""
    inp = stub_empty.get_inputs()[0]
    assert inp.shape[1] == 3
    assert inp.shape[2] == 640
    assert inp.shape[3] == 640


def test_no_detections_on_empty_output(stub_empty):
    """Пустой выход модели → пустой список детекций."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_people(stub_empty, frame)
    assert isinstance(result, list)
    assert result == []


def test_output_format(stub_one):
    """Формат выхода: список 5-tuple (x1,y1,x2,y2,conf) с float confidence."""
    frame = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
    result = detect_people(stub_one, frame)
    assert isinstance(result, list)
    assert len(result) >= 1
    for x1, y1, x2, y2, conf in result:
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
        assert x1 <= x2 and y1 <= y2


def test_low_conf_filtered():
    """Бокс с confidence ниже порога отфильтровывается."""
    sess = StubSession(boxes=[(320.0, 320.0, 100.0, 200.0, 0.10)])
    result = detect_people(sess, np.zeros((480, 640, 3), dtype=np.uint8))
    assert result == []


def test_draw_boxes_empty():
    """draw_boxes не падает на пустом списке детекций."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_boxes(frame, [])
    assert out.shape == frame.shape


def test_draw_boxes_with_detection():
    """draw_boxes не падает при наличии bbox."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_boxes(frame, [(50, 50, 200, 400, 0.91)])
    assert out.shape == frame.shape
