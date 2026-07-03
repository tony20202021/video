"""Adaptive rate limiter shared across pipeline scripts."""
from __future__ import annotations

import time


class AdaptiveRateLimiter:
    """Adjusts a work interval to keep work/sleep ratio within bounds.

    Usage:
        limiter = AdaptiveRateLimiter(min_interval=0.5, max_interval=30.0)
        while True:
            t0 = time.monotonic()
            do_work()
            work_ms = (time.monotonic() - t0) * 1000
            sleep_ms = limiter.sleep(t0)
            msg = limiter.adapt(work_ms, sleep_ms)
            if msg:
                print(msg)
    """

    def __init__(
        self,
        min_interval: float,
        max_interval: float,
        factor: float = 2.0,
        high: float = 0.90,
        low: float = 0.40,
        window: int = 10,
        label: str = "adaptive",
        unit: str = "/с",
    ) -> None:
        self.interval     = min_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.factor       = factor
        self.high         = high
        self.low          = low
        self.window       = window
        self.label        = label
        self.unit         = unit
        self._log: list[tuple[float, float]] = []  # (work_ms, sleep_ms)

    def sleep(self, work_start: float) -> float:
        """Sleep to fill the current interval since work_start. Returns sleep_ms."""
        if self.interval <= 0:
            return 0.0
        remaining = self.interval - (time.monotonic() - work_start)
        if remaining > 0:
            time.sleep(remaining)
            return remaining * 1000.0
        return 0.0

    def adapt(self, work_ms: float, sleep_ms: float) -> str | None:
        """Record timing; adjust interval if ratio out of bounds. Returns log line or None."""
        self._log.append((work_ms, sleep_ms))
        if self.min_interval <= 0 or len(self._log) < self.window:
            return None
        w     = self._log[-self.window:]
        work  = sum(r[0] for r in w)
        total = sum(r[0] + r[1] for r in w)
        if total <= 0:
            return None
        ratio = work / total
        if ratio > self.high and self.interval < self.max_interval:
            self.interval = min(self.interval * self.factor, self.max_interval)
            rate = (1.0 / self.interval) if self.interval > 0 else 0.0
            return f"  [{self.label}] {ratio:.0%} работы → ↓ {rate:.2f}{self.unit}"
        if ratio < self.low and self.interval > self.min_interval:
            self.interval = max(self.interval / self.factor, self.min_interval)
            rate = (1.0 / self.interval) if self.interval > 0 else 0.0
            return f"  [{self.label}] {ratio:.0%} работы → ↑ {rate:.2f}{self.unit}"
        return None

    @property
    def timing_log(self) -> list[tuple[float, float]]:
        return list(self._log)
