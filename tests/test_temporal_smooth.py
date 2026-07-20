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
