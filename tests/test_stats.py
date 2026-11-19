"""9.13 Evaluator and statistics."""
import numpy as np
import pytest
from scipy import stats as sps

from mosaic.eval.compare import compare, outcome
from mosaic.eval.stats import holm, mcnemar_exact, n_for_half_width, paired_bootstrap, wilson
from mosaic.types import RunRecord


def test_paired_bootstrap_known_answer():
    # constant difference: every resample has the same mean, so the CI collapses on it
    a = np.zeros(50)
    b = np.full(50, 100.0)
    assert paired_bootstrap(a, b) == (100.0, 100.0, 100.0)
    # large n: bootstrap one-sided 95% bound ~ mean - 1.645 * sd / sqrt(n)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 4000) * 100.0
    b = np.where(rng.random(4000) < 0.9, a, 100 - a)
    d = b - a
    mean, lo, hi = paired_bootstrap(a, b, n_boot=4000)
    expect = d.mean() - 1.645 * d.std(ddof=1) / np.sqrt(len(d))
    assert abs(lo - expect) < 0.4 and lo < mean < hi


@pytest.mark.parametrize("n10, n01", [(10, 2), (0, 0), (5, 5), (1, 12), (30, 18)])
def test_mcnemar_matches_scipy(n10, n01):
    a = np.array([False] * n10 + [True] * n01 + [True] * 7)
    b = np.array([True] * n10 + [False] * n01 + [True] * 7)
    got = mcnemar_exact(a, b)
    ref = 1.0 if n10 + n01 == 0 else sps.binomtest(min(n10, n01), n10 + n01, 0.5, alternative="two-sided").pvalue
    assert got[:2] == (n10, n01) and abs(got[2] - ref) < 1e-12
    if (n10, n01) == (10, 2):
        assert abs(got[2] - 0.03857) < 1e-4  # textbook value


def test_holm_correction():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5]) == [0.5]


def test_verdicts_and_reasons():
    assert outcome(1.0, 9.0, 5)[0] == "superior"
    assert outcome(-3.0, 4.0, 5)[0] == "non-inferior"
    assert outcome(-15.0, -7.0, 5)[0] == "inferior"
    o, why = outcome(-8.0, 2.0, 5)
    assert o == "inconclusive" and "straddles" in why


def rec(qid, cls, size, ok, pipe):
    return RunRecord(run_id=pipe, pipeline=pipe, question_id=qid, cls=cls, size=size, correct=ok)


def test_compare_pairs_by_question_and_applies_holm_across_cells():
    rng = np.random.default_rng(1)
    A, B = [], []
    for cls in "AP":
        for size in ("128k", "512k"):
            for i in range(60):
                qid = f"{cls}{size}{i}"
                a = bool(rng.random() < 0.3)
                A.append(rec(qid, cls, size, a, "P0"))
                B.append(rec(qid, cls, size, bool(rng.random() < 0.9), "P3"))
    res = compare(A, list(reversed(B)), margin=5, pooled={"primary": (("A", "P"), ("128k", "512k"))})
    assert len(res["cells"]) == 4 and all(v.p_adj >= v.p_value for v in res["cells"])
    prim = res["primary"][0]
    assert prim.n == 240 and prim.outcome == "superior"


def test_wilson_and_sample_size():
    lo, hi = wilson(54, 60)
    assert 0.79 < lo < 0.80 and 0.95 < hi < 0.96
    assert 900 <= n_for_half_width(0.10, 0.02) <= 1000
