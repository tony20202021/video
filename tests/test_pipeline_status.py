"""Тесты pipeline_status.py: склонение и формат ячейки «N прогонов (X кадров/файлов)»."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
import pipeline_status as ps  # noqa: E402


def test_plural_runs():
    p = lambda n: ps._plural(n, "прогон", "прогона", "прогонов")
    assert p(1) == "прогон"
    assert p(2) == "прогона"
    assert p(4) == "прогона"
    assert p(5) == "прогонов"
    assert p(11) == "прогонов"   # 11-14 — исключение
    assert p(21) == "прогон"
    assert p(22) == "прогона"


def test_plural_frames():
    p = lambda n: ps._plural(n, "кадр", "кадра", "кадров")
    assert p(0) == "кадров"
    assert p(1) == "кадр"
    assert p(262) == "кадра"     # …2 → few
    assert p(330) == "кадров"    # …0 → many


def _st(count, window, frames=0, bursts=0, **kw):
    d = {"count": count, "window": window, "frames": frames, "bursts": bursts,
         "min": None, "avg": None, "max": None, "last": None}
    d.update(kw)
    return d


def test_fmt_stats_batch():
    # yolo/classify/identify: N прогонов (X кадров) — X из лога «1 батч (X кадров)»
    assert ps.fmt_stats(_st(9, 60, frames=330), False, kind="batch") \
        == "9 прогонов (1ч) (330 кадров)"
    assert ps.fmt_stats(_st(1, 10, frames=1), False, kind="batch") \
        == "1 прогон (10м) (1 кадр)"
    # даже если ничего не найдено — прогоны считаются, кадры показываются
    assert ps.fmt_stats(_st(10, 10, frames=0), False, kind="batch") \
        == "10 прогонов (10м) (0 кадров)"


def test_fmt_stats_burst():
    # transfer: N всплесков приёма (прогонов) (X принятых файлов)
    assert ps.fmt_stats(_st(328, 10, bursts=12), False, kind="burst") \
        == "12 прогонов (10м) (328 файлов)"


def test_fmt_stats_timing_unit():
    cell = ps.fmt_stats(_st(5, 10, frames=100, min=3.4, avg=43.7, max=202.5, last=5.2),
                        has_timing=True, kind="batch")
    assert cell == "5 прогонов (10м) (100 кадров)\n1прогон 3.4с/43.7с/202.5с/5.2с"


def test_fmt_stats_zero():
    assert ps.fmt_stats(_st(0, 10), False, kind="batch") == "—"
