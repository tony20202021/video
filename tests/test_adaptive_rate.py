"""Тесты AdaptiveRateLimiter (src/common/utils/adaptive_rate.py)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from common.utils.adaptive_rate import AdaptiveRateLimiter


def _make(min_interval=0.2, max_interval=4.0, factor=2.0, high=0.90, low=0.40, window=3) -> AdaptiveRateLimiter:
    return AdaptiveRateLimiter(
        min_interval=min_interval,
        max_interval=max_interval,
        factor=factor,
        high=high,
        low=low,
        window=window,
    )


def _fill(limiter: AdaptiveRateLimiter, work_ms: float, sleep_ms: float, n: int) -> str | None:
    """Feed n identical timing samples; return last adapt() result."""
    msg = None
    for _ in range(n):
        msg = limiter.adapt(work_ms, sleep_ms)
    return msg


# ─── Initial state ────────────────────────────────────────────────────────────

def test_initial_interval_equals_min():
    lim = _make(min_interval=0.5, max_interval=10.0)
    assert lim.interval == 0.5


# ─── No adapt before window is full ──────────────────────────────────────────

def test_no_adapt_before_window():
    lim = _make(window=5)
    for _ in range(4):
        result = lim.adapt(900.0, 100.0)  # 90% work — above high
    assert result is None
    assert lim.interval == lim.min_interval  # unchanged


# ─── Slows down when overloaded ──────────────────────────────────────────────

def test_slows_down_when_overloaded():
    lim = _make(min_interval=0.2, max_interval=4.0, factor=2.0, high=0.80, window=3)
    msg = _fill(lim, work_ms=900.0, sleep_ms=100.0, n=3)  # 90% work > 80% high
    assert lim.interval == 0.4  # 0.2 * 2
    assert msg is not None
    assert "↓" in msg


def test_slows_down_respects_max_interval():
    lim = _make(min_interval=0.2, max_interval=0.3, factor=2.0, high=0.80, window=3)
    _fill(lim, work_ms=900.0, sleep_ms=100.0, n=3)
    assert lim.interval == 0.3  # clamped at max


# ─── Speeds up when underloaded ──────────────────────────────────────────────

def test_speeds_up_when_underloaded():
    lim = _make(min_interval=0.1, max_interval=1.6, factor=2.0, low=0.50, window=3)
    lim.interval = 0.4  # start above min
    msg = _fill(lim, work_ms=100.0, sleep_ms=900.0, n=3)  # 10% work < 50% low
    assert lim.interval == 0.2  # 0.4 / 2
    assert msg is not None
    assert "↑" in msg


def test_speeds_up_respects_min_interval():
    lim = _make(min_interval=0.2, max_interval=4.0, factor=2.0, low=0.50, window=3)
    lim.interval = 0.25  # just above min
    _fill(lim, work_ms=10.0, sleep_ms=990.0, n=3)
    assert lim.interval == 0.2  # clamped at min


# ─── No change within bounds ─────────────────────────────────────────────────

def test_no_change_within_bounds():
    lim = _make(high=0.90, low=0.40, window=3)
    msg = _fill(lim, work_ms=600.0, sleep_ms=400.0, n=3)  # 60% — in [40%, 90%]
    assert msg is None
    assert lim.interval == lim.min_interval


# ─── Already at boundary — no change ────────────────────────────────────────

def test_no_slowdown_at_max_interval():
    lim = _make(min_interval=0.2, max_interval=4.0, factor=2.0, high=0.80, window=3)
    lim.interval = 4.0  # already at max
    before = lim.interval
    _fill(lim, work_ms=900.0, sleep_ms=100.0, n=3)
    assert lim.interval == before


def test_no_speedup_at_min_interval():
    lim = _make(min_interval=0.2, max_interval=4.0, factor=2.0, low=0.50, window=3)
    # interval starts at min_interval = 0.2, already at min
    before = lim.interval
    _fill(lim, work_ms=10.0, sleep_ms=990.0, n=3)
    assert lim.interval == before


# ─── sleep() ─────────────────────────────────────────────────────────────────

def test_sleep_returns_zero_when_interval_is_zero():
    lim = _make(min_interval=0.0, max_interval=1.0)
    lim.interval = 0.0
    slept = lim.sleep(time.monotonic())
    assert slept == 0.0


def test_sleep_returns_zero_when_overtime():
    lim = _make(min_interval=0.01, max_interval=1.0)
    lim.interval = 0.01
    past = time.monotonic() - 1.0  # work started 1s ago, interval already passed
    slept = lim.sleep(past)
    assert slept == 0.0


def test_sleep_duration_approximate():
    lim = _make(min_interval=0.05, max_interval=1.0)
    lim.interval = 0.05
    t0 = time.monotonic()
    slept = lim.sleep(t0)
    elapsed = (time.monotonic() - t0) * 1000
    assert slept > 0
    assert elapsed >= 40  # at least ~40ms (allow generous tolerance)


# ─── Interval=0 skips adapt ──────────────────────────────────────────────────

def test_adapt_skips_when_min_interval_zero():
    """When min_interval=0, adapt never triggers — rate is unlimited."""
    lim = _make(min_interval=0.0, max_interval=1.0, window=3)
    msg = _fill(lim, work_ms=900.0, sleep_ms=100.0, n=5)
    assert msg is None


# ─── timing_log property ─────────────────────────────────────────────────────

def test_timing_log_accumulates():
    lim = _make(window=10)
    lim.adapt(100.0, 200.0)
    lim.adapt(150.0, 250.0)
    log = lim.timing_log
    assert len(log) == 2
    assert log[0] == (100.0, 200.0)
    assert log[1] == (150.0, 250.0)
