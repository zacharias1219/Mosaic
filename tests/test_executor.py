"""9.8 Executor (filter, chunk, map, reduce)."""
import asyncio
import json
import time

from mosaic.core.executor import Executor, apply_filter, chunk, parse_log, parse_rows
from mosaic.core.plan_schema import Plan, validate_plan
from mosaic.core.reducers import reduce_rows
from mosaic.tokens import count_tokens
from tests.helpers import ScriptedClient, cand, gateway


def plan(**kw):
    base = {"strategy": "recurse", "filter": {"keywords": ["REFUND", "money back"], "fields": {}, "semantic": True},
            "chunk_tokens": 1500, "map_task": "x", "map_schema": {"is_refund": "bool"}, "reduce": "count",
            "reduce_args": {"where": {"is_refund": True}}}
    base.update(kw)
    return validate_plan(base)


def echo_rows(c, messages):
    """Mapper that marks a ticket as refund when its text mentions a dollar sign."""
    lines = messages[-1]["content"].split("TICKETS:\n", 1)[1].split("\n")
    return json.dumps({"rows": [{"ticket_id": ln.split(" |")[0], "is_refund": "$" in ln} for ln in lines if ln]})


def test_keyword_filter_case_insensitive_and_logged(tickets_8k):
    recs = parse_log([t.line() for t in tickets_8k])
    kept, hits = apply_filter(recs, plan(filter={"keywords": ["REFUND"], "fields": {}, "semantic": False}))
    assert hits["refund"] == len(kept) == sum(1 for t in tickets_8k if "refund" in t.text.lower())
    kept2, hits2 = apply_filter(recs, plan())
    assert len(kept2) == len(recs) and hits2["refund"] == hits["refund"]  # semantic=true keeps everything


def test_field_filters_exact(tickets_8k):
    recs = parse_log([t.line() for t in tickets_8k])
    p = plan(filter={"fields": {"product": "Mesh Pro", "months": [3, 4]}, "keywords": [], "semantic": True})
    kept, _ = apply_filter(recs, p)
    assert kept and all(r.product == "Mesh Pro" and r.ts[5:7] in ("03", "04") for r in kept)
    assert len(kept) == sum(1 for t in tickets_8k if t.product == "Mesh Pro" and t.ts[5:7] in ("03", "04"))


def test_chunking_never_splits_a_ticket(tickets_8k):
    recs = parse_log([t.line() for t in tickets_8k])
    for size in (200, 700, 1500, 4000):
        chunks = chunk(recs, size)
        assert [r.ticket_id for c in chunks for r in c] == [r.ticket_id for r in recs]
        for c in chunks:
            assert len(c) == 1 or sum(count_tokens(r.line) + 1 for r in c) <= size


async def test_semantic_true_sends_every_ticket_to_the_mapper(tickets_8k):
    client = ScriptedClient(echo_rows)
    gw = gateway([cand("A")], client)
    lines = [t.line() for t in tickets_8k]
    res = await Executor(gw).run(plan(), lines, "q", "count", 0.0, "q1", None)
    sent = sum(c[1][-1]["content"].count("\n") for c in client.calls)
    assert sent >= len(lines) - len(client.calls)
    assert res.answer == {"count": sum(1 for t in tickets_8k if "$" in t.text)}


def test_deterministic_reducers_on_fixture_rows():
    rows = [{"ticket_id": "T1", "customer_id": "C1", "ts": "2026-01-01T00:00:00Z", "category": "shipping", "amt": 10.0},
            {"ticket_id": "T2", "customer_id": "C1", "ts": "2026-01-05T00:00:00Z", "category": "defect", "amt": None},
            {"ticket_id": "T3", "customer_id": "C2", "ts": "2026-01-01T00:00:00Z", "category": "Shipping", "amt": 2.5},
            {"ticket_id": "T4", "customer_id": "C2", "ts": "2026-02-01T00:00:00Z", "category": "defect", "amt": "$1.25"}]
    assert reduce_rows(rows, "count", {"where": {"category": "shipping"}}) == 2
    assert reduce_rows(rows, "sum", {"field": "amt", "where": {}}) == 13.75
    assert reduce_rows(rows, "set_union", {"field": "customer_id", "where": {"category": "defect"}}) == ["C1", "C2"]
    pj = {"field": "category", "a": "shipping", "b": "defect", "window_days": 7, "output": "ids"}
    assert reduce_rows(rows, "pair_join", pj) == ["C1"]
    assert reduce_rows(rows, "pair_join", {**pj, "window_days": 31, "output": "count"}) == 2


async def test_failed_map_call_retried_on_next_candidate_and_traced(tickets_8k):
    client = ScriptedClient(echo_rows, fail_ids={"CHEAP"})
    gw = gateway([cand("CHEAP", price_in=0.01, price_out=0.01), cand("PRICEY")], client)
    lines = [t.line() for t in tickets_8k[:10]]
    res = await Executor(gw).run(plan(chunk_tokens=50000), lines, "q", "count", 0.0, "q1", None)
    spans = gw.tracer.spans["q1"]
    failed = [s for s in spans if not s.ok]
    assert failed and failed[0].candidate_id == "CHEAP" and "falling back" in failed[0].note
    assert any(s.ok and s.candidate_id == "PRICEY" and s.retries == 1 for s in spans)
    assert res.answer is not None


async def test_malformed_map_output_falls_back(tickets_8k):
    client = ScriptedClient(lambda c, m: "oops" if c.id == "CHEAP" else echo_rows(c, m))
    gw = gateway([cand("CHEAP", price_in=0.01, price_out=0.01), cand("PRICEY")], client)
    res = await Executor(gw).run(plan(chunk_tokens=50000), [t.line() for t in tickets_8k[:8]], "q", "count", 0.0, "q", None)
    assert [c[0] for c in client.calls] == ["CHEAP", "PRICEY"] and res.answer is not None


async def test_recursion_splits_oversized_chunks_and_caps_depth(tickets_8k):
    client = ScriptedClient(echo_rows)
    small = cand("S", context_window=700)
    lines = [t.line() for t in tickets_8k[:24]]
    res = await Executor(gateway([small], client), max_depth=2).run(plan(chunk_tokens=50000), lines, "q", "count", 0.0, "q", None)
    assert res.depth_cap_hit and any("depth cap 2" in n for n in res.notes)
    client2 = ScriptedClient(echo_rows)
    res2 = await Executor(gateway([cand("S", context_window=3000)], client2), max_depth=2).run(
        plan(chunk_tokens=50000), lines, "q", "count", 0.0, "q", None)
    assert not res2.depth_cap_hit and len(client2.calls) > 1 and res2.answer["count"] == sum("$" in x for x in lines)


async def test_map_calls_concurrent_up_to_limit_and_slow_provider_does_not_block(tickets_8k):
    client = ScriptedClient(echo_rows, sleep_s=lambda c: 0.25 if c.id == "SLOW" else 0.05)
    slow, fast = cand("SLOW", max_concurrency=2), cand("FAST", max_concurrency=2)
    gw = gateway([slow, fast], client, real=True, explore_rate=0.5, seed=3)
    lines = [t.line() for t in tickets_8k[:40]]
    t0 = time.perf_counter()
    await Executor(gw).run(plan(chunk_tokens=300), lines, "q", "count", gw.clock.now(), "q", None)
    elapsed = time.perf_counter() - t0
    spans = [s for s in gw.tracer.spans["q"] if s.role == "mapper"]
    n_slow = sum(s.candidate_id == "SLOW" for s in spans)
    n_fast = len(spans) - n_slow
    assert n_slow and n_fast
    # each provider is held to 2 in flight, and the two run side by side
    assert client.max_active <= 4
    serial = n_slow * 0.25 + n_fast * 0.05
    assert elapsed < 0.75 * serial
    fast_ends = [s.end_ms for s in spans if s.candidate_id == "FAST"]
    slow_ends = [s.end_ms for s in spans if s.candidate_id == "SLOW"]
    assert min(fast_ends) < min(slow_ends)


def test_parse_rows_joins_header_metadata_and_ignores_unknown_ids(tickets_8k):
    recs = parse_log([t.line() for t in tickets_8k[:3]])
    text = json.dumps({"rows": [{"ticket_id": recs[0].ticket_id, "amt": "$5"}, {"ticket_id": "T-99999", "amt": 1}]})
    rows, bad = parse_rows(text, recs, {"amt": "amount"})
    assert not bad and rows == [{"ticket_id": recs[0].ticket_id, "ts": recs[0].ts, "customer_id": recs[0].customer_id,
                                 "product": recs[0].product, "amt": 5.0}]
    assert parse_rows("garbage", recs, {})[1]
