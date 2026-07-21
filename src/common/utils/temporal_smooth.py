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
                    gate: float | None = None,
                    guard_sec: float | None = None,
                    protect_class: str | None = None,
                    minority_classes: list[str] | None = None,
                    min_minority_prob: float | None = None,
                    context: list[dict] | None = None,
                    veto_prob: float | None = None) -> list[tuple[str, bool]]:
    """Сглаживает предсказания по времени. records: [{'cam', 't', 'probs': [по classes]}].
    Возвращает (smoothed_class, changed) в ПОРЯДКЕ ВХОДА, changed = класс изменился vs argmax модели.

    gate: уверенные (max prob ≥ gate) не трогаем.
    Три предохранителя против ложных флипов «разных людей в резидента» (revert флипа = вернуть argmax):
      • guard_sec — не флипать одиночный кадр, если в ±guard_sec НЕТ кадра с argmax = новый класс
        (одиночные кадры другого человека, случай 014723/164148);
      • min_minority_prob + protect_class/minority_classes — не давить в protect_class кадр, чья
        argmax = меньшинство с prob ≥ min_minority_prob (реальный переход, случай 120912);
      • context (кадры M>1, НЕ сглаживаются) + veto_prob — если в визите есть M>1-кадр с prob
        меньшинства ≥ veto_prob, не форсить M=1 в protect_class (гости в многолюдных, случай 194041)."""
    out: list[tuple[str, bool] | None] = [None] * len(records)
    protect_idx = classes.index(protect_class) if protect_class in classes else None
    minority_idx = {classes.index(c) for c in (minority_classes or []) if c in classes}

    ctx_by_cam: dict[str, list[dict]] = {}
    for r in (context or []):
        ctx_by_cam.setdefault(r.get("cam", ""), []).append(r)
    for c in ctx_by_cam:
        ctx_by_cam[c].sort(key=lambda r: r["t"])

    by_cam: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_cam.setdefault(r.get("cam", ""), []).append(i)
    for cam, idxs in by_cam.items():
        idxs.sort(key=lambda i: records[i]["t"])
        times = [records[i]["t"] for i in idxs]
        cam_ctx = ctx_by_cam.get(cam, [])
        for visit in segment_visits(times, gap_sec):
            vi = [idxs[j] for j in visit]
            P = np.array([records[i]["probs"] for i in vi], dtype=float)
            vtimes = [records[i]["t"] for i in vi]
            argmax = [int(P[pos].argmax()) for pos in range(len(vi))]
            path = viterbi(P, p_stay)
            # (M>1-контекст) есть ли в пределах времени визита многолюдный кадр с сильным меньшинством
            veto_minority = False
            if context is not None and veto_prob is not None and minority_idx:
                t0, t1 = vtimes[0], vtimes[-1]
                for r in cam_ctx:
                    if t0 - 1e-3 <= r["t"] <= t1 + 1e-3 and any(
                            r["probs"][mi] >= veto_prob for mi in minority_idx):
                        veto_minority = True
                        break
            for pos, i in enumerate(vi):
                model_idx = argmax[pos]
                sm_idx = path[pos]
                if gate is not None and P[pos].max() >= gate:
                    sm_idx = model_idx
                if sm_idx != model_idx:                      # флип — проверяем предохранители
                    revert = False
                    if guard_sec is not None:                # нет поддержки нового класса рядом
                        t = vtimes[pos]
                        revert = not any(argmax[q] == sm_idx and abs(vtimes[q] - t) <= guard_sec
                                         for q in range(len(vi)) if q != pos)
                    into_protect = protect_idx is not None and sm_idx == protect_idx
                    if (not revert and into_protect and min_minority_prob is not None
                            and model_idx in minority_idx and P[pos][model_idx] >= min_minority_prob):
                        revert = True                        # сильная улика меньшинства
                    if not revert and into_protect and veto_minority:
                        revert = True                        # M>1-контекст против резидента
                    if revert:
                        sm_idx = model_idx
                out[i] = (classes[sm_idx], sm_idx != model_idx)
    return [o if o is not None else (classes[0], False) for o in out]
