"""9.9 Router."""
from collections import Counter

from mosaic.providers.ratelimit import Limiter
from mosaic.routing.predictor import Model, Predictor
from mosaic.routing.router import PoolState, Router, rule_quality
from mosaic.types import CallSpec
from tests.helpers import cand

MAP = CallSpec(role="mapper", tokens_in_est=3000, tokens_out_est=400, on_critical_path=False,
               features={"n_tickets": 25, "has_category": 1.0})
PLAN = CallSpec(role="planner", tokens_in_est=2000, tokens_out_est=300, on_critical_path=True, features={})


def state(cands, at=0.0, limiters=None, exclude=None):
    cs = {c.id: c for c in cands}
    lims = limiters or {c.id: Limiter(c.rpm, c.tpm, c.max_concurrency, virtual=True) for c in cands}
    return PoolState(cs, lims, at, {}, exclude or set())


def pred(qs: dict[str, float]) -> Predictor:
    return Predictor({f"mapper|{k}": Model(base_rate=v) for k, v in qs.items()} |
                     {f"planner|{k}": Model(base_rate=v) for k, v in qs.items()})


def test_window_and_role_infeasible_never_chosen():
    tiny = cand("TINY", context_window=2000, price_in=0, price_out=0)
    norole = cand("NOROLE", roles=["planner"], price_in=0, price_out=0)
    ok = cand("OK")
    for mode in ("rules", "learned", "fixed"):
        r = Router(mode, fixed="TINY", predictor=pred({"TINY": 0.99, "NOROLE": 0.99, "OK": 0.95}))
        assert r.route(MAP, state([tiny, norole, ok])).candidate_id == "OK"


def test_cheapest_above_threshold_wins():
    a = cand("A", price_in=1.0, price_out=1.0)
    b = cand("B", price_in=0.1, price_out=0.1)
    c = cand("C", price_in=0.01, price_out=0.01)
    r = Router("learned", predictor=pred({"A": 0.99, "B": 0.93, "C": 0.80}))
    d = r.route(MAP, state([a, b, c]))
    assert d.candidate_id == "B" and not d.flagged
    # nobody clears the bar: highest predicted quality, flagged
    r2 = Router("learned", predictor=pred({"A": 0.85, "B": 0.6, "C": 0.5}))
    d2 = r2.route(MAP, state([a, b, c]))
    assert d2.candidate_id == "A" and d2.flagged


def test_critical_path_prefers_lowest_latency():
    slow_cheap = cand("SLOW", price_in=0.01, price_out=0.01, tps_out=50, ttft_ms=2000)
    fast = cand("FAST", price_in=1.0, price_out=1.0, tps_out=3000, ttft_ms=100)
    r = Router("learned", predictor=pred({"SLOW": 0.95, "FAST": 0.95}))
    assert r.route(PLAN, state([slow_cheap, fast])).candidate_id == "FAST"
    assert r.route(MAP, state([slow_cheap, fast])).candidate_id == "SLOW"


def test_rate_limited_candidate_skipped_while_headroom_elsewhere():
    rationed = cand("C4", price_in=0.01, price_out=0.01, rpm=5, tpm=30000)
    bulk = cand("C5", price_in=0.5, price_out=0.5)
    lims = {"C4": Limiter(5, 30000, 4, virtual=True), "C5": Limiter(450, 150000, 16, virtual=True)}
    r = Router("learned", predictor=pred({"C4": 0.97, "C5": 0.95}))
    picks = []
    for i in range(8):
        d = r.route(MAP, state([rationed, bulk], at=1000.0, limiters=lims))
        lims[d.candidate_id].reserve(3400, 1000.0)
        picks.append(d.candidate_id)
    assert picks[:5] == ["C4"] * 5 and set(picks[5:]) == {"C5"}


def test_exploration_rate_and_propensity_logged():
    a, b, c = cand("A", price_in=0.1), cand("B", price_in=0.2), cand("C", price_in=0.3)
    r = Router("learned", predictor=pred({"A": 0.95, "B": 0.95, "C": 0.95}), explore_rate=0.2, seed=1)
    ds = [r.route(MAP, state([a, b, c])) for _ in range(3000)]
    explored = [d for d in ds if d.explored]
    assert 0.17 < len(explored) / len(ds) < 0.23
    greedy = [d for d in ds if d.candidate_id == "A"]
    assert all(abs(d.propensity - (0.8 + 0.2 / 3)) < 1e-9 for d in greedy)
    assert all(abs(d.propensity - 0.2 / 3) < 1e-9 for d in ds if d.candidate_id != "A")
    assert Counter(d.candidate_id for d in explored).keys() == {"A", "B", "C"}


def test_predictor_disabled_falls_back_to_rule_table():
    big = cand("BIG", sim={"tier": "large"}, price_in=1.0, price_out=1.0)
    tiny = cand("TINY", sim={"tier": "tiny"}, price_in=0.0, price_out=0.0)
    r = Router("learned", predictor=None)
    d = r.route(MAP, state([big, tiny]))
    assert d.candidate_id == "BIG" and "[rules]" in d.reason
    assert rule_quality(tiny, MAP) < 0.9
    easy = CallSpec(role="mapper", tokens_in_est=500, tokens_out_est=60, on_critical_path=False,
                    features={"n_tickets": 3, "has_bool": 1.0})
    assert rule_quality(tiny, easy) >= 0.9


def test_latency_budget_drops_candidates_that_would_blow_it():
    cpu = cand("CPU", price_in=0, price_out=0, prefill_tps=60, tps_out=12, sim={"tier": "tiny"})
    c5 = cand("C5")
    r = Router("learned", predictor=pred({"CPU": 0.95, "C5": 0.95}), latency_budget_ms=15000)
    assert r.route(MAP, state([cpu, c5])).candidate_id == "C5"
