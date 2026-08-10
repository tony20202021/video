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

DEFAULT_GAP_SEC = 5.0      # пауза больше → новый визит
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


def window_average(probs: np.ndarray, times, window_sec: float,
                   triangular: bool = False, kmax: int | None = None) -> np.ndarray:
    """Классическое бегущее окно: для КАЖДОГО кадра i его вектор вероятностей заменяется на
    среднее по соседям в окне вокруг НЕГО (меняется только значение самого i; соседи берутся как
    есть). На краях отрезка окно одностороннее, но усредняются ВСЕ кадры.

    window_sec — ПОЛНЫЙ размер окна по времени (сек): радиус = window_sec/2 (напр. 3с → ±1.5с).
    kmax — ПОЛНОЕ макс. число кадров в окне: позиционный радиус = (kmax-1)//2 (напр. 7 → ±3 кадра).
      Окно = ПЕРЕСЕЧЕНИЕ ограничений (в пределах ±радиус_кадров И ±радиус_времени). kmax=None —
      без ограничения по числу кадров (только время). triangular: вес соседа ∝ (1-|dt|/радиус).
    Требует времена, отсортированные по возрастанию (визит сортируется по времени)."""
    P = np.asarray(probs, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(t)
    out = np.zeros_like(P)
    rt = window_sec / 2.0                              # радиус по времени
    rk = (kmax - 1) // 2 if kmax else None             # радиус по числу кадров (позиционный)
    for i in range(n):
        lo = 0 if rk is None else max(0, i - rk)
        hi = n if rk is None else min(n, i + rk + 1)
        j = np.arange(lo, hi)
        d = np.abs(t[j] - t[i])
        j = j[d <= rt]                                 # ∩ временно́е окно (±window_sec/2)
        if len(j) == 0:
            j = np.array([i])
        dd = np.abs(t[j] - t[i])
        w = (1.0 - dd / rt) if (triangular and rt > 0) else np.ones(len(j))
        w = np.clip(w, 1e-9, None)
        out[i] = (P[j] * w[:, None]).sum(0) / w.sum()
    return out


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
                    veto_prob: float | None = None,
                    prob_window_sec: float | None = None,
                    prob_window_tri: bool = False,
                    prob_window_kmax: int | None = None,
                    hard_vote: bool = False) -> list[tuple[str, bool]]:
    """Сглаживает предсказания по времени. records: [{'cam', 't', 'probs': [по classes]}].
    Возвращает (smoothed_class, changed) в ПОРЯДКЕ ВХОДА, changed = класс изменился vs argmax модели.

    hard_vote: если True — РЕЖИМ ЖЁСТКОГО ГОЛОСА (замер v3/v4: 11.6%/16.3% ошибок со склейкой зон,
    лучший из всех + бьёт baseline по recall всех 4 классов): на ВЕСЬ визит один класс = argmax
    среднего probs визита (Viterbi/окно/gate/предохранители НЕ применяются — приоритетный режим).
    БЕЗ склейки зон топит меньшинства (ЖКХ recall 48→35) — включать вместе с merge_zones.
    prob_window_sec: иначе если задан — РЕЖИМ БЕГУЩЕГО ОКНА (замер на 20260720: 8.0%→3.5%, лучше по ВСЕМ
    классам): внутри визита классическое бегущее окно (window_average) — probs каждого кадра →
    среднее соседей в ±window_sec/2 (и ≤ prob_window_kmax кадров), затем argmax
    (Viterbi/gate/предохранители НЕ применяются — они для HMM-режима).
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
            if hard_vote:                                # жёсткий голос: 1 класс на ВЕСЬ визит (argmax среднего probs)
                Pv = np.array([records[i]["probs"] for i in vi], dtype=float)
                cls_idx = int(Pv.mean(0).argmax())
                for pos, i in enumerate(vi):
                    out[i] = (classes[cls_idx], cls_idx != int(Pv[pos].argmax()))
                continue
            if prob_window_sec is not None:              # режим бегущего окна: усреднить probs → argmax
                Praw = np.array([records[i]["probs"] for i in vi], dtype=float)
                vtimes = [records[i]["t"] for i in vi]
                Pw = window_average(Praw, vtimes, prob_window_sec, prob_window_tri, prob_window_kmax)
                for pos, i in enumerate(vi):
                    sm_idx = int(Pw[pos].argmax())
                    out[i] = (classes[sm_idx], sm_idx != int(Praw[pos].argmax()))
                continue
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
