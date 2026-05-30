"""Детекция людей через YOLOv8n ONNX."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

YOLO_INPUT_SIZE = 640
PERSON_CLASS = 0
_BOX_COLOR = (0, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_model(model_path: Path):
    """Загружает ONNX-модель. Возвращает сессию или None при ошибке."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Нужен onnxruntime: pip install onnxruntime", file=sys.stderr)
        return None
    if not model_path.is_file():
        print(
            f"Модель не найдена: {model_path}\n"
            f"  Скачать: python scripts/setup_models.py",
            file=sys.stderr,
        )
        return None
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_x, pad_y


def _postprocess(
    output: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    nms_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    preds = output[0].T  # [8400, 84]
    person_scores = preds[:, 4 + PERSON_CLASS]
    mask = person_scores >= conf_threshold
    if not mask.any():
        return []

    scores = person_scores[mask]
    bxywh = preds[mask, :4]
    cx, cy, bw, bh = bxywh[:, 0], bxywh[:, 1], bxywh[:, 2], bxywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2

    indices = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, bw, bh], axis=1).tolist(),
        scores.tolist(),
        conf_threshold,
        nms_threshold,
    )

    result = []
    for i in (indices.flatten() if len(indices) else []):
        rx1 = int((float(x1[i]) - pad_x) / scale)
        ry1 = int((float(y1[i]) - pad_y) / scale)
        rx2 = int((float(x1[i]) + float(bxywh[i, 2]) - pad_x) / scale)
        ry2 = int((float(y1[i]) + float(bxywh[i, 3]) - pad_y) / scale)
        rx1 = max(0, min(rx1, orig_w - 1))
        ry1 = max(0, min(ry1, orig_h - 1))
        rx2 = max(0, min(rx2, orig_w))
        ry2 = max(0, min(ry2, orig_h))
        result.append((rx1, ry1, rx2, ry2, float(scores[i])))
    return result


def detect_people(
    sess,
    bgr: np.ndarray,
    *,
    conf_threshold: float = 0.35,
    nms_threshold: float = 0.45,
) -> list[tuple[int, int, int, int, float]]:
    """Возвращает список (x1, y1, x2, y2, confidence) для людей."""
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    output = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    return _postprocess(
        output,
        orig_w=w, orig_h=h,
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )


def draw_boxes(bgr: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 2), (x1 + tw, y1), _BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - bl - 1), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
