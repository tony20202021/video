"""Мульти-лейбл: формат labels.json и утилиты (классы <-> мультихот, пороги).

Модель 1 (группы) переходит с single-label (softmax+argmax, 1 класс на кроп) на MULTI-LABEL:
в кроп YOLO (bbox+паддинг) часто попадает несколько людей РАЗНЫХ классов — кроп размечается
и предсказывается НАБОРОМ классов (sigmoid + порог на каждый класс).

Формат labels.json:
  v2 (мульти):  {"version": 2, "task": "...", "labels": {img: ["class1", "class2"]}}
  v1 (старый):  {"version": 1, "labels": {img: "class"}}   ← читается как {img: ["class"]}
Пустой список [] = размечено «ни одного класса»; отсутствие ключа = ещё не размечено.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
LABELS_VERSION = 2


def normalize_label(val) -> list:
    """Значение метки → список классов. Старый формат (строка 'class') или новый (список)."""
    if not val:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple, set)):
        return [str(c) for c in val if c]
    return []


def load_labels(path) -> dict:
    """labels.json → {img: [classes]}. Поддерживает старый v1 ({img:'class'}) и новый v2
    ({img:[classes]}), а также список записей [{image, classes|class}]."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and "labels" in raw:      # обёртка {version, labels}
        raw = raw["labels"]
    if isinstance(raw, dict):
        return {k: normalize_label(v) for k, v in raw.items()}
    if isinstance(raw, list):                          # [{image, classes|class}]
        out = {}
        for r in raw:
            if isinstance(r, dict) and r.get("image"):
                out[r["image"]] = normalize_label(r.get("classes", r.get("class")))
        return out
    return {}


def save_labels(path, labels: dict, task: str = "classify") -> None:
    """Пишет {img: [classes]} в labels.json v2 ({version, task, labels}); классы сортируются, дубли убраны."""
    norm = {k: sorted(set(normalize_label(v))) for k, v in labels.items()}
    data = {"version": LABELS_VERSION, "task": task, "labels": norm}
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def to_multihot(classes, all_classes) -> list:
    """Список классов → мультихот-вектор 0/1 по порядку all_classes."""
    s = set(classes)
    return [1 if c in s else 0 for c in all_classes]


def from_multihot(vec, all_classes) -> list:
    """Мультихот-вектор → список классов (где 1)."""
    return [c for c, v in zip(all_classes, vec) if v]


def classes_from_probs(prob_map: dict, thresholds=DEFAULT_THRESHOLD) -> list:
    """{class: prob} + пороги (float для всех или {class: thr}) → классы с prob ≥ порога,
    отсортированные по убыванию вероятности (пустой список = ни одного класса выше порога)."""
    def _thr(cls: str) -> float:
        if isinstance(thresholds, dict):
            return float(thresholds.get(cls, DEFAULT_THRESHOLD))
        return float(thresholds)
    present = [(c, p) for c, p in prob_map.items() if p >= _thr(c)]
    present.sort(key=lambda cp: cp[1], reverse=True)
    return [c for c, _ in present]
