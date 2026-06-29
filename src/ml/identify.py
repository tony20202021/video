"""Идентификация жителя: MobileNetV3-Small классификатор.

Классы — конкретные жители (person_01, person_02, ...).
Модель обучается на кропах с тех же камер (ракурс сверху, разная одежда).
Стартовые веса: backbone от Модели 1 (GroupClassifier) — см. docs/ml.md.

Если модель не загружена — возвращает (None, 0.0) без ошибки.
Обучение: python scripts/train/train_resident.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 224
# ImageNet mean/std — тот же стандарт, что и у Модели 1
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class PersonIdentifier:
    """MobileNetV3-Small классификатор жителей.

    Классы хранятся в метаданных ONNX-модели (ключ "classes", JSON-список).
    При отсутствии метаданных используется список, переданный в конструктор.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self._sess = None
        self._input_name: str = "input"
        self._classes: list[str] = list(classes) if classes else []
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
        try:
            import json
            meta = self._sess.get_modelmeta()
            if "classes" in meta.custom_metadata_map:
                self._classes = json.loads(meta.custom_metadata_map["classes"])
        except Exception:
            pass
        return True

    @property
    def ready(self) -> bool:
        return self._sess is not None and bool(self._classes)

    @property
    def person_count(self) -> int:
        return len(self._classes)

    def identify(
        self,
        bgr_crop: np.ndarray,
        threshold: float = 0.70,
    ) -> tuple[str | None, float]:
        """Возвращает (person_id, confidence) или (None, best_conf) если ниже порога.

        Если модель не загружена или нет классов — (None, 0.0).
        """
        if not self.ready:
            return None, 0.0
        blob = _preprocess(bgr_crop)
        raw = self._sess.run(None, {self._input_name: blob})[0][0]
        probs = _softmax(raw)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf >= threshold:
            return self._classes[idx], conf
        return None, conf


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR crop → NCHW float32, нормализованный по ImageNet."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
