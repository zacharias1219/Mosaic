"""Statistics for paired comparisons (PRD 10)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 42, one_sided: bool = True) -> tuple[float, float, float]:
    """Bootstrap CI on mean(b - a) over paired items. One-sided 95% lower bound uses the 5th
    percentile (and the matching upper 95th), as non-inferiority needs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    q = alpha if one_sided else alpha / 2
    return float(d.mean()), float(np.quantile(means, q)), float(np.quantile(means, 1 - q))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar test on discordant pairs. Returns (b_only, a_only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(b & ~a))
    n01 = int(np.sum(a & ~b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(sps.binomtest(min(n10, n01), n, 0.5).pvalue)
    return n10, n01, p


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def ci_half_width(p_disc: float, n: int) -> float:
    return 1.96 * math.sqrt(p_disc / n) if n else float("inf")


def n_for_half_width(p_disc: float, half_width: float) -> int:
    return math.ceil((1.96 / half_width) ** 2 * p_disc)
