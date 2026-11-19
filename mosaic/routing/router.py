"""Router: drop what can't work, predict the rest, keep what clears the bar, then pick by cost
(off the critical path) or by latency (on it) (PRD 8, 9.9).

R0 fixed, R1 rules and R2 learned differ only in where the quality estimate comes from.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional

from mosaic.providers.local import cpu_cost
from mosaic.providers.ratelimit import Limiter
from mosaic.routing.predictor import Predictor
from mosaic.tokens import TOKEN_MARGIN
from mosaic.types import CallSpec, Candidate, Decision, Profile

# R1: what a sensible engineer would write without data. Keyed by model tier.
RULES: dict[str, dict[str, float]] = {
    "large": {"planner": 0.96, "mapper": 0.95, "aggregator": 0.96, "direct": 0.95},
    "mid": {"planner": 0.92, "mapper": 0.93, "aggregator": 0.92, "direct": 0.93},
    "small": {"planner": 0.85, "mapper": 0.90, "aggregator": 0.86, "direct": 0.88},
    "tiny": {"planner": 0.40, "mapper": 0.60, "aggregator": 0.50, "direct": 0.70},
}


def rule_quality(cand: Candidate, spec: CallSpec) -> float:
    tier = cand.sim.get("tier", "mid")
    q = RULES[tier][spec.role]
    f = spec.features
    if tier == "tiny":  # tiny model only for small, simple chunks or short direct reads
        if spec.role == "mapper" and f.get("n_tickets", 99) <= 6 and not f.get("has_category") and not f.get("has_amount"):
            q = 0.92
        if spec.role == "direct" and spec.tokens_in_est <= 1200:
            q = 0.91
    if spec.role == "direct" and spec.tokens_in_est > 32_000:
        q -= 0.25  # long-context degradation is common knowledge
    return q


@dataclass
class PoolState:
    candidates: dict[str, Candidate]
    limiters: dict[str, Limiter]
    at: float
    profiles: dict[str, Profile] = field(default_factory=dict)
    exclude: set[str] = field(default_factory=set)
    local_cost: dict[str, float] = field(default_factory=lambda: {"watts": 28, "usd_per_kwh": 0.10,
                                                                   "amortisation_usd_per_hour": 0.05})


def predict_latency(c: Candidate, spec: CallSpec, prof: Optional[Profile], wait_ms: float = 0.0) -> float:
    ttft = prof.ttft_base if prof else c.ttft_ms
    prefill = prof.prefill_tps if prof else c.prefill_tps
    tps = prof.tps_out if prof else c.tps_out
    out = spec.tokens_out_est + c.sim.get("reasoning_tokens", 0)
    return ttft + 1000 * spec.tokens_in_est / prefill + 1000 * out / tps + wait_ms


def predict_cost(c: Candidate, spec: CallSpec, latency_ms: float, local: dict[str, float]) -> float:
    if c.provider == "local":
        return cpu_cost(latency_ms / 1000, c.watts or local["watts"], local["usd_per_kwh"],
                        local["amortisation_usd_per_hour"])
    out = spec.tokens_out_est + c.sim.get("reasoning_tokens", 0)
    return (spec.tokens_in_est * c.price_in + out * c.price_out) / 1e6


class Router:
    def __init__(self, mode: Literal["fixed", "rules", "learned"] = "rules", fixed: Optional[str] = None,
                 predictor: Optional[Predictor] = None, thresholds: Optional[dict[str, float]] = None,
                 explore_rate: float = 0.0, seed: int = 42, latency_budget_ms: float = 15_000,
                 max_queue_ms: float = 2_000, pool: Optional[list[str]] = None):
        self.mode, self.fixed, self.predictor = mode, fixed, predictor
        self.tau = {"planner": 0.9, "mapper": 0.9, "aggregator": 0.9, "direct": 0.9, **(thresholds or {})}
        self.explore_rate = explore_rate
        self.rng = random.Random(f"router-{seed}")
        self.latency_budget_ms = latency_budget_ms
        self.max_queue_ms = max_queue_ms
        self.pool = pool

    def quality(self, c: Candidate, spec: CallSpec) -> tuple[float, str]:
        if self.mode == "learned" and self.predictor is not None:
            q = self.predictor.predict(spec.role, spec.features, c.id)
            if q is not None:
                return q, "learned"
        return rule_quality(c, spec), "rules"

    def feasible(self, spec: CallSpec, state: PoolState) -> list[tuple[Candidate, float]]:
        out = []
        need = spec.tokens_in_est + spec.tokens_out_est
        for c in state.candidates.values():
            if c.id in state.exclude or spec.role not in c.roles or (self.pool and c.id not in self.pool):
                continue
            if (need + c.sim.get("reasoning_tokens", 0)) * TOKEN_MARGIN > c.context_window:
                continue
            if c.max_call_tokens and need + c.sim.get("reasoning_tokens", 0) > c.max_call_tokens:
                continue
            lim = state.limiters.get(c.id)
            wait = lim.wait_ms(need, state.at) if lim else 0.0
            if wait == float("inf"):
                continue
            out.append((c, wait))
        return out

    def route(self, spec: CallSpec, state: PoolState) -> Optional[Decision]:
        F = self.feasible(spec, state)
        if not F:
            return None
        if self.mode == "fixed":
            hit = [x for x in F if x[0].id == self.fixed]
            c, wait = hit[0] if hit else F[0]
            lat = predict_latency(c, spec, state.profiles.get(c.id), wait)
            return Decision(candidate_id=c.id, q_hat=rule_quality(c, spec), l_hat=lat,
                            c_hat=predict_cost(c, spec, lat, state.local_cost), propensity=1.0,
                            reason="fixed" if hit else f"fixed candidate {self.fixed} infeasible; first feasible",
                            flagged=not hit)
        scored = []
        for c, wait in F:
            q, src = self.quality(c, spec)
            lat = predict_latency(c, spec, state.profiles.get(c.id), wait)
            scored.append((c, q, lat, predict_cost(c, spec, lat, state.local_cost), src, wait))
        # latency budget: drop candidates that alone would blow the per-question budget
        within = [s for s in scored if s[2] <= self.latency_budget_ms]
        scored = within or scored
        ok = [s for s in scored if s[1] >= self.tau[spec.role]]
        # capacity: among candidates that clear the bar, skip one that would queue while another has headroom.
        # A queue never pushes a call onto a candidate below the bar; waiting is better than a wrong answer.
        quick = [s for s in ok if s[5] <= self.max_queue_ms]
        ok = quick or ok
        key = (lambda s: (s[2], s[3])) if spec.on_critical_path else (lambda s: (s[3], s[2]))
        if ok:
            best, flagged, why = min(ok, key=key), False, "lowest latency above bar" if spec.on_critical_path else "cheapest above bar"
        else:
            best, flagged, why = max(scored, key=lambda s: (s[1], -s[2])), True, "no candidate clears bar; highest q"
        chosen, explored = best, False
        prop = 1.0
        if self.explore_rate > 0:
            if self.rng.random() < self.explore_rate:
                chosen, explored = self.rng.choice(scored), True
            k = len(scored)
            prop = self.explore_rate / k + ((1 - self.explore_rate) if chosen[0].id == best[0].id else 0.0)
        c, q, lat, cost, src, _ = chosen
        return Decision(candidate_id=c.id, q_hat=q, l_hat=lat, c_hat=cost, propensity=prop,
                        reason=("explore" if explored else why) + f" [{src}]", explored=explored, flagged=flagged)
