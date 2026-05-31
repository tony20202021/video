"""Полный ML-пайплайн для одного кадра: классификация → идентификация.

Использование:
    pipeline = MLPipeline.from_config(config)
    results = pipeline.run(bgr_frame, detections)
    # results: list[PersonResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.classify import GroupClassifier
from ml.identify import PersonIdentifier

CLASSES = ["resident", "courier", "delivery", "utilities", "other"]

# Пороги по умолчанию — переопределяются через MLConfig
_DEFAULT_CLASSIFY_THRESH = 0.65
_DEFAULT_IDENTIFY_THRESH = 0.75


@dataclass
class PersonResult:
    bbox: tuple[int, int, int, int]        # x1, y1, x2, y2
    detect_conf: float
    group_class: str                        # resident/courier/... / unknown
    group_conf: float
    group_probs: dict[str, float] = field(default_factory=dict)
    person_id: str | None = None
    identify_conf: float = 0.0
    identify_method: str = ""              # face | body | none


@dataclass
class MLConfig:
    classify_model: Path | None = None
    identify_model: Path | None = None
    classify_threshold: float = _DEFAULT_CLASSIFY_THRESH
    identify_threshold: float = _DEFAULT_IDENTIFY_THRESH
    crop_pad: float = 0.10                 # отступ при вырезке кропа


class MLPipeline:
    def __init__(self, config: MLConfig) -> None:
        self._cfg = config
        self._classifier = GroupClassifier(config.classify_model)
        self._identifier = PersonIdentifier(config.identify_model)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MLPipeline":
        """Создаёт пайплайн из словаря конфига (из config.yaml)."""
        models = cfg.get("models", {})
        thresholds = cfg.get("thresholds", {})
        return cls(MLConfig(
            classify_model=Path(models["classify"]) if "classify" in models else None,
            identify_model=Path(models["identify"]) if "identify" in models else None,
            classify_threshold=thresholds.get("classification", {}).get(
                "min_confidence", _DEFAULT_CLASSIFY_THRESH
            ),
            identify_threshold=thresholds.get("identification", {}).get(
                "min_confidence", _DEFAULT_IDENTIFY_THRESH
            ),
        ))

    @property
    def classifier(self) -> GroupClassifier:
        return self._classifier

    @property
    def identifier(self) -> PersonIdentifier:
        return self._identifier

    def load_person_embeddings(self, data: dict[str, list[list[float]]]) -> None:
        """Передаёт эмбеддинги жителей в идентификатор."""
        self._identifier.load_embeddings(data)

    def run(
        self,
        bgr_frame: np.ndarray,
        detections: list[tuple[int, int, int, int, float]],
    ) -> list[PersonResult]:
        """Обрабатывает все bounding boxes на кадре.

        detections: [(x1, y1, x2, y2, conf), ...]
        Возвращает список PersonResult — по одному на каждый bbox.
        """
        h, w = bgr_frame.shape[:2]
        results: list[PersonResult] = []

        for x1, y1, x2, y2, det_conf in detections:
            crop = _extract_crop(bgr_frame, x1, y1, x2, y2, self._cfg.crop_pad, w, h)

            # Классификация группы
            group_class, group_conf, group_probs = self._classifier.classify(crop)

            # Идентификация только для жителей (или если классификатор не готов)
            person_id: str | None = None
            id_conf: float = 0.0
            id_method: str = "none"

            should_identify = (
                self._identifier.ready
                and self._identifier.person_count > 0
                and (
                    group_class == "resident"
                    or not self._classifier.ready
                )
            )
            if should_identify:
                person_id, id_conf = self._identifier.identify(
                    crop, threshold=self._cfg.identify_threshold
                )
                id_method = "body"  # face detection отдельный модуль (RetinaFace)

            results.append(PersonResult(
                bbox=(x1, y1, x2, y2),
                detect_conf=det_conf,
                group_class=group_class,
                group_conf=group_conf,
                group_probs=group_probs,
                person_id=person_id,
                identify_conf=id_conf,
                identify_method=id_method,
            ))

        return results


def _extract_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    pad: float,
    w: int, h: int,
) -> np.ndarray:
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1c = max(0, x1 - px)
    y1c = max(0, y1 - py)
    x2c = min(w, x2 + px)
    y2c = min(h, y2 + py)
    if x2c <= x1c or y2c <= y1c:
        return frame[y1:y2, x1:x2].copy()
    return frame[y1c:y2c, x1c:x2c].copy()
