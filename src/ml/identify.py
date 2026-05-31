"""Идентификация жителя по эмбеддингу лица/тела.

Модель: MobileFaceNet ONNX (предобученная, без дообучения).
Метод: cosine similarity с предвычисленными эмбеддингами жителей из БД.

Скачать модель: python scripts/setup_models.py --task identify
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 112  # стандарт MobileFaceNet


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2_norm(a), _l2_norm(b)))


class PersonIdentifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self._sess = None
        self._input_name: str = "input"
        # person_id → усреднённый нормированный эмбеддинг
        self._embeddings: dict[str, np.ndarray] = {}
        if model_path is not None:
            self.load(Path(model_path))

    def load(self, model_path: Path) -> bool:
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

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray | None:
        """Вычисляет L2-нормированный эмбеддинг. None если модель не загружена."""
        if not self.ready:
            return None
        blob = _preprocess(bgr_crop)
        raw = self._sess.run(None, {self._input_name: blob})[0][0]
        return _l2_norm(raw)

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        """Добавляет эмбеддинг жителя. При повторном вызове — усредняет."""
        emb = _l2_norm(embedding)
        if person_id in self._embeddings:
            self._embeddings[person_id] = _l2_norm(
                self._embeddings[person_id] + emb
            )
        else:
            self._embeddings[person_id] = emb

    def load_embeddings(self, data: dict[str, list[list[float]]]) -> None:
        """Загружает словарь {person_id: [embedding_list, ...]} из БД.

        Каждая запись — список вектор-эмбеддингов (float) для одного жителя.
        Усредняем их в один вектор.
        """
        self._embeddings.clear()
        for pid, embs in data.items():
            if not embs:
                continue
            arr = np.mean(np.array(embs, dtype=np.float32), axis=0)
            self._embeddings[pid] = _l2_norm(arr)

    def identify(
        self,
        bgr_crop: np.ndarray,
        threshold: float = 0.75,
    ) -> tuple[str | None, float]:
        """Возвращает (person_id, similarity) или (None, best_sim) ниже порога."""
        emb = self.embed(bgr_crop)
        if emb is None or not self._embeddings:
            return None, 0.0
        best_id, best_sim = None, -1.0
        for pid, stored in self._embeddings.items():
            sim = float(np.dot(emb, stored))
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    @property
    def person_count(self) -> int:
        return len(self._embeddings)


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR crop → NCHW float32 в диапазоне [-1, 1] (стандарт MobileFaceNet)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) - 127.5) / 128.0
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
