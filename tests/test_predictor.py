"""9.10 Predictor trainer."""
import numpy as np

from mosaic.routing.features import FEATURES
from mosaic.routing.predictor import brier, reliability, train


def synth(n, seed, signal=True):
    rng = np.random.default_rng(seed)
    rows = []
    for cid, skill in (("C4", 2.0), ("C5", 1.0)):
        for _ in range(n):
            f = {k: float(rng.random()) for k in FEATURES["mapper"]}
            f["n_tickets"] = float(rng.integers(1, 40))
            z = skill - (0.08 * f["n_tickets"] + 1.5 * f["uncued_money"] if signal else 0)
            rows.append({"role": "mapper", "candidate_id": cid, "features": f,
                         "correct": bool(rng.random() < 1 / (1 + np.exp(-z)))})
    return rows


def test_logistic_per_role_candidate_is_inspectable_and_beats_constant():
    p = train(synth(600, 1), seed=1)
    rep = p.report["mapper|C5"]
    assert rep["kind"] == "logistic" and rep["beats_constant"]
    assert rep["coef"]["n_tickets"] < 0  # more tickets in a chunk, less likely all correct
    assert len(rep["reliability"]) <= 10 and all({"bin", "n", "mean_pred", "observed"} <= set(b) for b in rep["reliability"])
    q = p.predict("mapper", {"n_tickets": 2}, "C4")
    assert 0 < q < 1 and q > p.predict("mapper", {"n_tickets": 35}, "C4")


def test_reports_honestly_when_no_signal():
    p = train(synth(300, 2, signal=False), seed=2)
    rep = p.report["mapper|C5"]
    # with pure noise an overfit regression loses to the base rate, and the report says so
    assert rep["beats_constant"] is False and rep["brier"] >= rep["brier_constant"]


def test_small_samples_fall_back_to_base_rate():
    rows = synth(10, 3)
    p = train(rows, seed=3)
    assert p.report["mapper|C4"]["kind"] == "constant"
    assert abs(p.predict("mapper", {}, "C4") - np.mean([r["correct"] for r in rows if r["candidate_id"] == "C4"])) < 1e-9


def test_training_deterministic_for_fixed_seed(tmp_path):
    a, b = train(synth(400, 4), seed=7), train(synth(400, 4), seed=7)
    a.save(tmp_path / "a.json")
    b.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()


def test_brier_and_reliability_helpers():
    p = np.array([0.1, 0.9, 0.9, 0.1])
    y = np.array([0, 1, 1, 0])
    assert abs(brier(p, y) - 0.01) < 1e-12
    rel = reliability(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 1, 1, 1]))
    assert rel[0]["observed"] == 0.5 and rel[-1]["observed"] == 1.0
