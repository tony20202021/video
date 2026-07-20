"""Темпоральное сглаживание предсказаний классификатора групп (Модель 1) по времени.

Идея: последовательные кропы ОДНОЙ камеры в пределах «визита» — обычно один человек/событие,
класс меняется редко. Сглаживание убирает единичные ошибки модели, сохраняя реальные визиты.

Метод (по экспериментам на 9.4k размеченных кропов — лучший баланс точность/сохранение визитов):
  1. группировка по камере, сортировка по времени;
  2. сегментация потока на ВИЗИТЫ по паузам (пауза > gap_sec → новый визит);
  3. HMM / Viterbi ВНУТРИ визита: матрица переходов (p_stay на диагонали — «человек не
     телепортируется между классами»), эмиссия = вероятности модели. Класс может меняться
     внутри визита (в отличие от «1 класс на визит»), поэтому короткая доставка в потоке
     резидентов НЕ затирается;
  4. опциональный гейтинг: уверенные предсказания модели (max prob ≥ gate) не трогаем.

Замер (v3_1, 9 дат): baseline 25.7% ошибок → visit-HMM ~18%, recall доставки 0.68→0.69
(сохраняется), резидента 0.75→0.83. «1 класс на визит» даёт 11.6% ошибок, но роняет доставку
до 0.53 — поэтому нужен именно HMM внутри визита.
"""
from __future__ import annotations

import numpy as np

DEFAULT_GAP_SEC = 60.0     # пауза больше → новый визит
DEFAULT_P_STAY = 0.95      # вероятность сохранения класса между соседними кадрами визита


def segment_visits(times, gap_sec: float = DEFAULT_GAP_SEC) -> list[list[int]]:
    """Отсортированные времена → список визитов (списки индексов). Пауза > gap_sec — новый визит."""
    n = len(times)
    if n == 0:
        return []
    visits = [[0]]
    for i in range(1, n):
        if times[i] - times[i - 1] > gap_sec:
            visits.append([])
        visits[-1].append(i)
    return visits


def viterbi(probs: np.ndarray, p_stay: float = DEFAULT_P_STAY) -> list[int]:
    """probs: [n, K] вероятности по классам. Возвращает индексы наиболее вероятной
    последовательности состояний (Viterbi). Матрица переходов: p_stay на диагонали,
    (1-p_stay)/(K-1) на остальных."""
    probs = np.asarray(probs, dtype=float)
    n, K = probs.shape
    if n == 1 or K == 1:
        return [int(probs[t].argmax()) for t in range(n)]
    trans = np.full((K, K), (1.0 - p_stay) / (K - 1))
    np.fill_diagonal(trans, p_stay)
    logT = np.log(trans + 1e-12)
    E = probs / (probs.sum(1, keepdims=True) + 1e-9)     # нормируем в распределение
    logE = np.log(E + 1e-9)
    V = np.zeros((n, K))
    B = np.zeros((n, K), dtype=int)
    V[0] = logE[0] + np.log(1.0 / K)
    for t in range(1, n):
        for k in range(K):
            sc = V[t - 1] + logT[:, k]
            B[t, k] = int(sc.argmax())
            V[t, k] = sc.max() + logE[t, k]
    path = [int(V[-1].argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(B[t, path[-1]]))
    return list(reversed(path))


def smooth_sequence(records: list[dict], classes: list[str], *,
                    gap_sec: float = DEFAULT_GAP_SEC, p_stay: float = DEFAULT_P_STAY,
                    gate: float | None = None) -> list[tuple[str, bool]]:
    """Сглаживает предсказания по времени. records: [{'cam', 't', 'probs': [по classes]}].
    Возвращает список (smoothed_class, changed) в ПОРЯДКЕ ВХОДА, где changed = класс изменился
    относительно argmax модели. gate: если задан, уверенные (max prob ≥ gate) оставляем как есть."""
    out: list[tuple[str, bool] | None] = [None] * len(records)
    by_cam: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_cam.setdefault(r.get("cam", ""), []).append(i)
    for _cam, idxs in by_cam.items():
        idxs.sort(key=lambda i: records[i]["t"])
        times = [records[i]["t"] for i in idxs]
        for visit in segment_visits(times, gap_sec):
            vi = [idxs[j] for j in visit]
            P = np.array([records[i]["probs"] for i in vi], dtype=float)
            path = viterbi(P, p_stay)
            for pos, i in enumerate(vi):
                model_idx = int(P[pos].argmax())
                sm_idx = path[pos]
                if gate is not None and P[pos].max() >= gate:
                    sm_idx = model_idx
                out[i] = (classes[sm_idx], sm_idx != model_idx)
    return [o if o is not None else (classes[0], False) for o in out]
