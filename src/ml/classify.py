"""Классификация группы человека: 1_resident / 2_delivery / 3_utilities / 4_guest.

Модель: MobileNetV3-Small, обученная на кропах людей.
Вход:  BGR кроп произвольного размера → ресайз до 224×224
Выход: (class_name, confidence)

Если модель не найдена — возвращает ("unknown", 0.0) без ошибки.
Скачать/обучить: python scripts/train/train_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from common.utils.classes import GROUP_CLASSES as CLASSES
from common.utils.multilabel import DEFAULT_THRESHOLD, classes_from_probs
INPUT_SIZE = 224
# ImageNet mean/std — стандарт для MobileNetV3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GroupClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self._sess = None
        self._input_name: str = "input"
        # multi-label: sigmoid + порог на класс. Флаг и пороги — из манифеста рядом с моделью.
        self.multi_label: bool = False
        self.thresholds: dict[str, float] = {}
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
        self._load_manifest(model_path)
        return True

    def _load_manifest(self, model_path: Path) -> None:
        """Манифест рядом с моделью (v3_1.onnx → v3_1.json): multi_label + пороги по классам."""
        self.multi_label = False
        self.thresholds = {}
        mp = model_path.with_suffix(".json")
        if mp.is_file():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                self.multi_label = bool(m.get("multi_label", False))
                self.thresholds = {k: float(v) for k, v in (m.get("thresholds") or {}).items()}
            except Exception:
                pass

    @property
    def ready(self) -> bool:
        return self._sess is not None

    def _prob_map(self, bgr_crop: np.ndarray) -> dict[str, float]:
        """{class: prob}. multi_label → sigmoid (независимые), иначе softmax."""
        blob = _preprocess(bgr_crop)
        raw = self._sess.run(None, {self._input_name: blob})[0][0]
        probs = _sigmoid(raw) if self.multi_label else _softmax(raw)
        return {cls: float(p) for cls, p in zip(CLASSES, probs)}

    def classify(
        self, bgr_crop: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Обратно совместимо: (лучший class_name, его вероятность, {class: prob}).
        prob_map — softmax (single-label) или sigmoid (multi-label).
        Если модель не загружена — ("unknown", 0.0, {})."""
        if not self.ready:
            return "unknown", 0.0, {}
        prob_map = self._prob_map(bgr_crop)
        best = max(prob_map, key=prob_map.get)
        return best, prob_map[best], prob_map

    def predict_classes(
        self, bgr_crop: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        """Multi-label: (список классов, {class: prob}).
        multi_label модель → классы с sigmoid ≥ порога (пусто = ни одного, → uncertain);
        single-label модель → [argmax] (1 класс, порог уверенности решает вызывающий)."""
        if not self.ready:
            return [], {}
        prob_map = self._prob_map(bgr_crop)
        if self.multi_label:
            thr = self.thresholds or DEFAULT_THRESHOLD
            return classes_from_probs(prob_map, thr), prob_map
        best = max(prob_map, key=prob_map.get)
        return [best], prob_map


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR crop → NCHW float32, нормализованный по ImageNet."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
