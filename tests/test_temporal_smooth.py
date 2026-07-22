"""Тесты темпорального сглаживания: сегментация визитов, Viterbi, sequence-сглаживание."""
from __future__ import annotations

import numpy as np

from common.utils import temporal_smooth as ts

C = ["1_resident", "2_delivery", "3_utilities", "4_guest"]


def _p(cls, conf=0.9):
    """Вероятностный вектор с пиком на cls."""
    v = np.full(len(C), (1 - conf) / (len(C) - 1))
    v[C.index(cls)] = conf
    return v.tolist()


def test_segment_visits():
    # паузы: [0,1,2] близко, потом пауза >60 → новый визит
    times = [0, 5, 10, 100, 105]
    assert ts.segment_visits(times, gap_sec=60) == [[0, 1, 2], [3, 4]]
    assert ts.segment_visits([], 60) == []
    assert ts.segment_visits([42], 60) == [[0]]


def test_viterbi_fixes_isolated_error():
    # поток резидентов с одной ошибкой в середине → Viterbi чинит
    probs = np.array([_p("1_resident"), _p("1_resident"),
                      _p("2_delivery", 0.6),           # единичная ошибка
                      _p("1_resident"), _p("1_resident")])
    path = ts.viterbi(probs, p_stay=0.95)
    assert [C[i] for i in path] == ["1_resident"] * 5   # ошибка исправлена


def test_viterbi_keeps_real_run():
    # реальный визит доставки (несколько подряд) НЕ затирается
    probs = np.array([_p("1_resident"), _p("2_delivery"), _p("2_delivery"),
                      _p("2_delivery"), _p("1_resident")])
    path = [C[i] for i in ts.viterbi(probs, p_stay=0.9)]
    assert path.count("2_delivery") >= 3                 # доставка сохранена


def test_smooth_sequence_per_camera_and_visit():
    # две камеры; в cam_d поток резидентов с 1 ошибкой; сглаживание её чинит
    recs = [
        {"cam": "cam_d", "t": 0.0, "probs": _p("1_resident")},
        {"cam": "cam_d", "t": 1.0, "probs": _p("4_guest", 0.55)},   # ошибка
        {"cam": "cam_d", "t": 2.0, "probs": _p("1_resident")},
        {"cam": "cam_u", "t": 0.5, "probs": _p("2_delivery")},      # др. камера — не влияет
    ]
    out = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95)
    assert out[0][0] == "1_resident"
    assert out[1] == ("1_resident", True)      # исправлено (changed=True)
    assert out[2][0] == "1_resident"
    assert out[3][0] == "2_delivery"           # другая камера сохранена


def test_smooth_sequence_gate_keeps_confident():
    # уверенная (0.95) ошибка при gate=0.8 остаётся; без гейта — чинится
    recs = [
        {"cam": "c", "t": 0.0, "probs": _p("1_resident")},
        {"cam": "c", "t": 1.0, "probs": _p("4_guest", 0.95)},      # уверенно
        {"cam": "c", "t": 2.0, "probs": _p("1_resident")},
    ]
    gated = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95, gate=0.8)
    assert gated[1][0] == "4_guest"            # уверенное не тронуто
    nogate = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95)
    assert nogate[1][0] == "1_resident"        # без гейта — сглажено


def test_smooth_sequence_gap_splits_visits():
    # большая пауза между кадрами → разные визиты, сглаживание не «перетекает»
    recs = [
        {"cam": "c", "t": 0.0, "probs": _p("2_delivery")},
        {"cam": "c", "t": 1.0, "probs": _p("2_delivery")},
        {"cam": "c", "t": 500.0, "probs": _p("1_resident")},       # новый визит (пауза 499с)
        {"cam": "c", "t": 501.0, "probs": _p("1_resident")},
    ]
    out = [o[0] for o in ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.99)]
    assert out == ["2_delivery", "2_delivery", "1_resident", "1_resident"]


def test_smooth_sequence_guard_reverts_unsupported_flip():
    # одиночный гость с паузой 6с до пачки резидентов: без guard флипается в резидента,
    # с guard(5с) — нет поддержки резидента рядом → остаётся гостем (кейс 014723-типа)
    recs = [{"cam": "c", "t": 0.0, "probs": _p("4_guest", 0.6)}]
    recs += [{"cam": "c", "t": 6.0 + i * 0.1, "probs": _p("1_resident")} for i in range(10)]
    no_guard = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95)
    assert no_guard[0][0] == "1_resident"                      # без guard задавлен
    with_guard = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95, guard_sec=5.0)
    assert with_guard[0] == ("4_guest", False)                # guard вернул argmax


def test_smooth_sequence_min_minority_protects_transition():
    # сильная доставка на границе резидентского прогона: min_minority_prob не даёт задавить в res
    recs = [{"cam": "c", "t": float(i), "probs": _p("1_resident")} for i in range(4)]
    recs.append({"cam": "c", "t": 4.0, "probs": _p("2_delivery", 0.85)})   # сильная улика
    base = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95)
    assert base[-1][0] == "1_resident"                        # без защиты — заглажен в res
    prot = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95,
                              protect_class="1_resident", minority_classes=["2_delivery"],
                              min_minority_prob=0.7)
    assert prot[-1][0] == "2_delivery"                        # защита сохранила доставку


def test_window_average_denoises_spike():
    # поток резидентов с одним скачком в доставку → окно усредняет, argmax снова резидент
    P = np.array([_p("1_resident"), _p("1_resident"),
                  _p("2_delivery", 0.6), _p("1_resident"), _p("1_resident")])
    Pw = ts.window_average(P, [0, 1, 2, 3, 4], window_sec=3)
    assert int(Pw[2].argmax()) == C.index("1_resident")   # единичный скачок усреднён


def test_smooth_sequence_prob_window_fixes_error():
    # режим бегущего окна чинит единичную ошибку усреднением вероятностей
    recs = [{"cam": "c", "t": 0.0, "probs": _p("1_resident")},
            {"cam": "c", "t": 1.0, "probs": _p("2_delivery", 0.55)},   # ошибка
            {"cam": "c", "t": 2.0, "probs": _p("1_resident")}]
    out = ts.smooth_sequence(recs, C, gap_sec=60, prob_window_sec=3.0)
    assert out[1] == ("1_resident", True)                 # окно исправило (changed=True)


def test_window_average_kmax_limits_neighbors():
    # 5 кадров в 0.4с; полное окно давит доставку в резидента, kmax=2 (сам+ближайший) — сохраняет
    P = np.array([_p("2_delivery"), _p("2_delivery"), _p("1_resident"),
                  _p("1_resident"), _p("1_resident")])
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    full = ts.window_average(P, times, 3.0)
    k2 = ts.window_average(P, times, 3.0, kmax=2)
    assert int(full[0].argmax()) == C.index("1_resident")   # полное: 3 резид > 2 достав
    assert int(k2[0].argmax()) == C.index("2_delivery")     # K=2: сам+сосед-доставка


def test_prob_window_respects_visit_gap():
    # окно не усредняет через паузу визита (>gap) — разные события не смешиваются
    recs = [{"cam": "c", "t": 0.0, "probs": _p("2_delivery")},
            {"cam": "c", "t": 500.0, "probs": _p("1_resident")}]
    out = [o[0] for o in ts.smooth_sequence(recs, C, gap_sec=60, prob_window_sec=30.0)]
    assert out == ["2_delivery", "1_resident"]


def test_smooth_sequence_mgt1_context_veto():
    # M=1 кадр в визите, где M>1-контекст даёт сильного гостя → не форсим в резидента
    recs = [{"cam": "c", "t": 0.0, "probs": _p("4_guest", 0.6)},
            {"cam": "c", "t": 1.0, "probs": _p("1_resident")}]
    ctx = [{"cam": "c", "t": 0.5, "probs": _p("4_guest", 0.9)}]   # многолюдный кадр — гости
    no_veto = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95)
    assert no_veto[0][0] == "1_resident"                      # без вето — задавлен
    veto = ts.smooth_sequence(recs, C, gap_sec=60, p_stay=0.95,
                              protect_class="1_resident", minority_classes=["4_guest"],
                              context=ctx, veto_prob=0.85)
    assert veto[0] == ("4_guest", False)                      # вето по M>1-контексту
