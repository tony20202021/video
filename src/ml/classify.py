"""Классификация группы человека: 1_resident / 2_delivery / 3_utilities / 99_other.

Модель: MobileNetV3-Small, обученная на кропах людей.
Вход:  BGR кроп произвольного размера → ресайз до 224×224
Выход: (class_name, confidence)

Если модель не найдена — возвращает ("unknown", 0.0) без ошибки.
Скачать/обучить: python scripts/train/train_classifier.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ["1_resident", "2_delivery", "3_utilities", "99_other"]
INPUT_SIZE = 224
# ImageNet mean/std — стандарт для MobileNetV3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class GroupClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self._sess = None
        self._input_name: str = "input"
        if model_path is not None:
            self.load(Path(model_path))

    def load(self, model_path: Path) -> bool:
        """Загружает ONNX. Возвращает True при успехе."""
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        if not model_path.is_file():
            return False
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        return True

    @property
    def ready(self) -> bool:
        return self._sess is not None

    def classify(
        self, bgr_crop: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Возвращает (class_name, confidence, {class: prob}).

        Если модель не загружена — ("unknown", 0.0, {}).
        """
        if not self.ready:
            return "unknown", 0.0, {}
        blob = _preprocess(bgr_crop)
        raw = self._sess.run(None, {self._input_name: blob})[0][0]
        probs = _softmax(raw)
        idx = int(np.argmax(probs))
        prob_map = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
        return CLASSES[idx], float(probs[idx]), prob_map


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR crop → NCHW float32, нормализованный по ImageNet."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
