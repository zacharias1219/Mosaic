"""Per-candidate admission control: requests and tokens per minute (sliding 60 s window),
daily caps and concurrency (PRD 9.4, 13).

The same limiter runs on a real clock (it sleeps) or in virtual time (it only computes the
start time), so simulated runs see exactly the queueing that live runs would.
"""
from __future__ import annotations

import asyncio
import bisect
import time
from typing import Optional

WINDOW_MS = 60_000.0
DAY_MS = 86_400_000.0


class QuotaExhausted(Exception):
    pass


class Limiter:
    def __init__(self, rpm: Optional[int] = None, tpm: Optional[int] = None, max_concurrency: int = 8,
                 rpd: Optional[int] = None, tpd: Optional[int] = None, virtual: bool = True,
                 safety: float = 1.0):
        self.rpm = int(rpm * safety) if rpm else None
        self.tpm = int(tpm * safety) if tpm else None
        self.rpd, self.tpd = rpd, tpd
        self.max_concurrency = max_concurrency
        self.virtual = virtual
        self.events: list[tuple[float, int]] = []          # (start_ms, tokens), kept sorted
        self._max_at = 0.0
        self.intervals: list[list[float]] = []              # [start_ms, end_ms] (virtual mode)
        self.day_requests = 0
        self.day_tokens = 0
        self.day = 0
        self._sem = asyncio.Semaphore(max_concurrency) if not virtual else None
        self._lock = asyncio.Lock() if not virtual else None

    # ---- feasibility -------------------------------------------------------------------
    def _roll(self, at: float) -> None:
        d = int(at // DAY_MS)
        if d > self.day:
            self.day, self.day_requests, self.day_tokens = d, 0, 0

    def fits(self, tokens: int, at: Optional[float] = None) -> bool:
        """Could a call of this size ever be admitted (today)?"""
        if at is not None:
            self._roll(at)
        if self.tpm and tokens > self.tpm:
            return False
        if self.rpd and self.day_requests >= self.rpd:
            return False
        if self.tpd and self.day_tokens + tokens > self.tpd:
            return False
        return True

    def _near(self, lo: float, hi: float) -> list[tuple[float, int]]:
        i = bisect.bisect_right(self.events, (lo, float("inf")))
        j = bisect.bisect_left(self.events, (hi, -1))
        return self.events[i:j]

    def _window_ok(self, t: float, tokens: int) -> bool:
        near = self._near(t - WINDOW_MS, t + WINDOW_MS)
        anchors = [s for s, _ in near if s <= t] + [t]
        for w in anchors:  # every 60 s window that would contain the new event
            inside = [(s, k) for s, k in near if w <= s < w + WINDOW_MS]
            if self.rpm and len(inside) + 1 > self.rpm:
                return False
            if self.tpm and sum(k for _, k in inside) + tokens > self.tpm:
                return False
        return True

    def _conc_ok(self, t: float) -> bool:
        if not self.virtual:
            return True
        busy = sum(1 for s, e in self.intervals if s <= t < e)
        return busy < self.max_concurrency

    def earliest(self, tokens: int, at: float) -> float:
        """Earliest start time >= at (ms) at which the call would be admitted."""
        self._roll(at)
        if self.tpm and tokens > self.tpm:
            raise QuotaExhausted(f"call of {tokens} tokens can never be admitted")
        if not self.fits(tokens, at):  # today's quota is spent: the earliest slot is tomorrow
            if self.virtual:
                at = (self.day + 1) * DAY_MS
                self._roll(at)
            else:
                raise QuotaExhausted("daily quota spent")
        cands = sorted({at} | {s + WINDOW_MS for s, _ in self._near(at - WINDOW_MS, float("inf"))}
                       | {e for _, e in self.intervals if e > at})
        for t in cands:
            if self._window_ok(t, tokens) and self._conc_ok(t):
                return t
        return cands[-1] + WINDOW_MS  # unreachable in practice

    def wait_ms(self, tokens: int, at: float) -> float:
        try:
            return self.earliest(tokens, at) - at
        except QuotaExhausted:
            return float("inf")

    def headroom(self, at: float) -> float:
        """Share of the request budget left in the current window (1.0 = idle)."""
        if not self.rpm:
            return 1.0
        used = sum(1 for s, _ in self._near(at - WINDOW_MS, at + 1e-9))
        return max(0.0, 1 - used / self.rpm)

    # ---- admission ----------------------------------------------------------------------
    def reserve(self, tokens: int, at: float) -> float:
        start = self.earliest(tokens, at)
        self.record(start, tokens)
        self.day_requests += 1
        self.day_tokens += tokens
        return start

    def record(self, start: float, tokens: int) -> None:
        bisect.insort(self.events, (start, tokens))
        self._max_at = max(self._max_at, start)
        if len(self.events) > 4000:  # events far behind the newest can no longer matter
            cut = bisect.bisect_left(self.events, (self._max_at - 30 * WINDOW_MS, -1))
            del self.events[:cut]

    def finish(self, start: float, end: float) -> None:
        if self.virtual:
            self.intervals.append([start, end])
            if len(self.intervals) > 500:
                self.intervals = [iv for iv in self.intervals if iv[1] > self._max_at - 30 * WINDOW_MS]

    async def acquire_real(self, tokens: int, clock: "RealClock") -> float:
        assert self._lock is not None and self._sem is not None
        async with self._lock:
            start = self.reserve(tokens, clock.now())
        delay = start - clock.now()
        if delay > 0:
            await asyncio.sleep(delay / 1000)
        await self._sem.acquire()
        return clock.now()

    def release_real(self) -> None:
        if self._sem is not None:
            self._sem.release()


class RealClock:
    virtual = False

    def __init__(self):
        self._t0 = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self._t0) * 1000


class VirtualClock:
    """Time only moves when the pipeline says so; every call carries its own issue time."""
    virtual = True

    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def advance_to(self, t: float) -> None:
        self.t = max(self.t, t)
