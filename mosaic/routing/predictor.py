"""Quality predictor: P(call correct | features, candidate), one logistic regression per
(role, candidate), calibrated if needed (PRD 8, 9.10)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mosaic.routing.features import FEATURES, vector

MIN_ROWS = 30


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)) if len(y) else float("nan")


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.sum():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                        "observed": float(y[m].mean()), "gap": float(abs(p[m].mean() - y[m].mean()))})
    return out


@dataclass
class Model:
    """Either a fitted LR (coef, intercept, scaling) or a constant base rate."""
    base_rate: float
    coef: Optional[list[float]] = None
    intercept: float = 0.0
    mean: Optional[list[float]] = None
    scale: Optional[list[float]] = None
    iso_x: Optional[list[float]] = None
    iso_y: Optional[list[float]] = None
    n: int = 0

    def predict(self, x: list[float]) -> float:
        if self.coef is None:
            return self.base_rate
        z = (np.asarray(x) - np.asarray(self.mean)) / np.asarray(self.scale)
        p = float(1 / (1 + np.exp(-(z @ np.asarray(self.coef) + self.intercept))))
        if self.iso_x:
            p = float(np.interp(p, self.iso_x, self.iso_y))
        return min(max(p, 1e-4), 1 - 1e-4)


@dataclass
class Predictor:
    models: dict[str, Model] = field(default_factory=dict)  # key "role|candidate"
    report: dict[str, Any] = field(default_factory=dict)

    def predict(self, role: str, feats: dict[str, float], candidate: str) -> Optional[float]:
        m = self.models.get(f"{role}|{candidate}")
        return None if m is None else m.predict(vector(role, feats))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"models": {k: m.__dict__ for k, m in self.models.items()},
                                          "report": self.report}, indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        d = json.loads(Path(path).read_text())
        return cls({k: Model(**v) for k, v in d["models"].items()}, d.get("report", {}))


def _fit(X: np.ndarray, y: np.ndarray, seed: int) -> Model:
    base = float(y.mean())
    if len(y) < MIN_ROWS or y.min() == y.max() or X.shape[1] == 0:
        return Model(base_rate=base, n=len(y))
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit((X - mean) / scale, y)
    return Model(base_rate=base, coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), mean=mean.tolist(),
                 scale=scale.tolist(), n=len(y))


def train(rows: list[dict], seed: int = 42, holdout: float = 0.3, calib_gap: float = 0.05) -> Predictor:
    """rows: {role, candidate_id, features, correct}. Splits per key into train / held-out,
    reports Brier against a constant predictor, and applies isotonic scaling if the
    held-out reliability gap is large and there is enough data."""
    rng = np.random.default_rng(seed)
    keys = sorted({(r["role"], r["candidate_id"]) for r in rows})
    pred, rep = Predictor(), {}
    for role, cand in keys:
        if role not in FEATURES:
            continue
        sub = [r for r in rows if r["role"] == role and r["candidate_id"] == cand]
        X = np.array([vector(role, r["features"]) for r in sub], float).reshape(len(sub), len(FEATURES[role]))
        y = np.array([1.0 if r["correct"] else 0.0 for r in sub])
        idx = rng.permutation(len(sub))
        n_test = int(len(sub) * holdout)
        te, tr = idx[:n_test], idx[n_test:]
        m = _fit(X[tr], y[tr], seed)
        p_te = np.array([m.predict(x) for x in X[te]])
        const = np.full(len(te), y[tr].mean() if len(tr) else 0.5)
        rel = reliability(p_te, y[te])
        max_gap = max((b["gap"] for b in rel if b["n"] >= 10), default=0.0)
        calibrated = False
        if m.coef is not None and max_gap > calib_gap and len(tr) >= 3 * MIN_ROWS:
            # isotonic on out-of-fold predictions of the training part
            half = len(tr) // 2
            m_a = _fit(X[tr[:half]], y[tr[:half]], seed)
            pa = np.array([m_a.predict(x) for x in X[tr[half:]]])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pa, y[tr[half:]])
            m.iso_x, m.iso_y = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
            p_cal = np.array([m.predict(x) for x in X[te]])
            if brier(p_cal, y[te]) <= brier(p_te, y[te]):
                p_te, rel, calibrated = p_cal, reliability(p_cal, y[te]), True
            else:
                m.iso_x = m.iso_y = None
        m_full = _fit(X, y, seed)  # final model on all rows, keeping the calibration decision
        m_full.iso_x, m_full.iso_y = m.iso_x, m.iso_y
        pred.models[f"{role}|{cand}"] = m_full
        b, bc = brier(p_te, y[te]), brier(const, y[te])
        rep[f"{role}|{cand}"] = {
            "n_train": int(len(tr)), "n_test": int(len(te)), "base_rate": float(y.mean()),
            "brier": b, "brier_constant": bc, "beats_constant": bool(b < bc), "calibrated_isotonic": calibrated,
            "max_bin_gap": float(max((x["gap"] for x in rel if x["n"] >= 10), default=0.0)),
            "reliability": rel, "kind": "logistic" if m_full.coef is not None else "constant",
            "coef": dict(zip(FEATURES[role], m_full.coef)) if m_full.coef else {},
        }
    pred.report = rep
    return pred
