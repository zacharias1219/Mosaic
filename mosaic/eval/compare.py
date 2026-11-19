"""Paired, per-class non-inferiority comparisons between two runs (PRD 9.13, 10)."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from mosaic.eval.stats import holm, mcnemar_exact, paired_bootstrap
from mosaic.types import RunRecord, Verdict

WORDING = {"non-inferior": "No worse than the baseline within {m} points", "superior": "Better than the baseline",
           "inferior": "Worse than the baseline", "inconclusive": "Not resolved by this sample"}


def outcome(ci_low: float, ci_high: float, margin: float) -> tuple[str, str]:
    """Fixed in advance (PRD 10). Superiority is checked first because it implies non-inferiority."""
    if ci_low > 0:
        return "superior", f"CI lower bound {ci_low:+.1f} > 0"
    if ci_low > -margin:
        return "non-inferior", f"CI lower bound {ci_low:+.1f} > -{margin:g}"
    if ci_high < -margin:
        return "inferior", f"CI upper bound {ci_high:+.1f} < -{margin:g}"
    return "inconclusive", f"CI [{ci_low:+.1f}, {ci_high:+.1f}] straddles -{margin:g}"


def pair_up(a: Iterable[RunRecord], b: Iterable[RunRecord]) -> list[tuple[RunRecord, RunRecord]]:
    bm = {r.question_id: r for r in b}
    return [(r, bm[r.question_id]) for r in a if r.question_id in bm]


def verdict(pairs: list[tuple[RunRecord, RunRecord]], margin: float, cls: str, size: str,
            n_boot: int = 10_000, seed: int = 42) -> Verdict:
    ya = np.array([p[0].correct for p in pairs], float)
    yb = np.array([p[1].correct for p in pairs], float)
    diff, lo, hi = paired_bootstrap(ya * 100, yb * 100, n_boot=n_boot, seed=seed)
    _, _, p = mcnemar_exact(ya.astype(bool), yb.astype(bool))
    out, why = outcome(lo, hi, margin)
    return Verdict(cls=cls, size=size, n=len(pairs), acc_a=float(ya.mean()) if len(ya) else 0.0,
                   acc_b=float(yb.mean()) if len(yb) else 0.0, diff=diff, ci_low=lo, ci_high=hi, p_value=p,
                   margin=margin, outcome=out, why=why)


def compare(runs_a: list[RunRecord], runs_b: list[RunRecord], margin: float = 5.0,
            classes: tuple[str, ...] = ("L", "A", "P"), pooled: dict[str, tuple] | None = None,
            n_boot: int = 10_000, seed: int = 42) -> dict[str, list[Verdict]]:
    """Per-(class, size) cells (exploratory, Holm-adjusted) plus named pooled analyses, e.g.
    pooled={"primary": (("A", "P"), ("128k", "512k"))}."""
    pairs = pair_up(runs_a, runs_b)
    cells: dict[tuple[str, str], list] = defaultdict(list)
    for pa, pb in pairs:
        cells[(pa.cls, pa.size)].append((pa, pb))
    order = {"8k": 0, "32k": 1, "128k": 2, "512k": 3}
    keys = sorted((k for k in cells if k[0] in classes), key=lambda k: (k[0], order.get(k[1], 9)))
    per_cell = [verdict(cells[k], margin, k[0], k[1], n_boot, seed) for k in keys]
    for v, padj in zip(per_cell, holm([v.p_value for v in per_cell])):
        v.p_adj = padj
    out = {"cells": per_cell}
    for name, (cls_set, sizes) in (pooled or {}).items():
        sel = [p for p in pairs if p[0].cls in cls_set and p[0].size in sizes]
        if sel:
            out[name] = [verdict(sel, margin, "+".join(cls_set), "+".join(sizes), n_boot, seed)]
    return out
