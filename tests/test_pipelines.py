"""9.11 Pipelines and 9.12 Tracer."""
import asyncio
import json
import time

from mosaic.data.questions import build_questions
from mosaic.gateway import Gateway
from mosaic.pipelines import RunContext, read_run, run_pipeline, write_run
from mosaic.providers.ratelimit import RealClock, VirtualClock
from mosaic.providers.sim import SimClient
from mosaic.routing.router import Router
from mosaic.trace import Tracer, critical_path_ms, question_cost, validate_nesting
from mosaic.types import RunRecord
from tests.helpers import ScriptedClient, cand

POOL = None


def pool():
    return {
        "C4": cand("C4", provider="cerebras", rpm=5, tpm=30000, sim={"tier": "large", "weights": "big", "malformed": 0,
                                                                   "error_rate": 0}),
        "C5": cand("C5", provider="cerebras", sim={"tier": "mid", "weights": "mid", "error_rate": 0}),
        "C6": cand("C6", provider="local", roles=["mapper", "direct"], context_window=8192, max_concurrency=1,
                   prefill_tps=90, tps_out=14, price_in=0, price_out=0, sim={"tier": "tiny", "weights": "tiny"}),
    }


def ctx_for(pipe, world, tickets, mode, fixed="C5", window=None):
    cands = pool()
    if window:
        cands["C5"] = cands["C5"].model_copy(update={"context_window": window})
    router = Router(mode, fixed=fixed, pool=list(cands))
    sim = SimClient(world)
    gw = Gateway(cands, {k: sim for k in cands}, router, Tracer(pipe), VirtualClock())
    return RunContext(gw, pipe, {"8k": tickets}, planner="C5")


def questions(world):
    return sorted(world.questions.values(), key=lambda q: q.question_id)[::4]


async def test_every_pipeline_emits_same_schema(world_8k, tickets_8k):
    qs = questions(world_8k)
    keys = None
    for pipe, mode in (("P0", "fixed"), ("P1", "fixed"), ("P2", "rules"), ("P3", "rules")):
        recs = await run_pipeline(pipe, qs, ctx_for(pipe, world_8k, tickets_8k, mode))
        assert len(recs) == len(qs)
        for r in recs:
            d = json.loads(r.model_dump_json())
            keys = keys or set(d)
            assert set(d) == keys and RunRecord(**d) == r


async def test_p0_records_truncation(world_8k, tickets_8k):
    qs = [q for q in questions(world_8k) if q.cls == "A"]
    recs = await run_pipeline("P0", qs, ctx_for("P0", world_8k, tickets_8k, "fixed", window=4000))
    assert all(r.truncated for r in recs) and all(any("truncated" in n for n in r.notes) for r in recs)


async def test_p1_and_p2_differ_only_in_routing(world_8k, tickets_8k):
    qs = [q for q in questions(world_8k) if q.cls != "L"]
    r1 = await run_pipeline("P1", qs, ctx_for("P1", world_8k, tickets_8k, "fixed"))
    r2 = await run_pipeline("P2", qs, ctx_for("P2", world_8k, tickets_8k, "rules"))
    for a, b in zip(r1, r2):
        assert a.plan == b.plan
        ca = [s.features for s in a.spans if s.role == "mapper" and s.ok]
        cb = [s.features for s in b.spans if s.role == "mapper" and s.ok]
        assert sorted(map(str, ca)) == sorted(map(str, cb))  # same chunks
    used1 = {s.candidate_id for r in r1 for s in r.spans if s.role == "mapper" and s.ok}
    assert used1 == {"C5"}


async def test_spans_nest_and_costs_sum(world_8k, tickets_8k):
    recs = await run_pipeline("P2", questions(world_8k), ctx_for("P2", world_8k, tickets_8k, "rules"))
    for r in recs:
        assert validate_nesting(r.spans)
        assert abs(question_cost(r.spans) - r.cost) < 1e-12
        roles = [s.role for s in r.spans]
        assert roles[0] == "question" and "planner" in roles and "filter" in roles


async def test_critical_path_matches_wall_clock_within_5_percent(tickets_8k):
    from mosaic.core.plan_schema import validate_plan
    from mosaic.core.prompts import MAP_SYSTEM
    plan = {"strategy": "recurse", "filter": {"keywords": [], "fields": {}, "semantic": True}, "chunk_tokens": 1500,
            "map_task": "x", "map_schema": {"is_refund": "bool"}, "reduce": "count",
            "reduce_args": {"where": {"is_refund": True}}}

    def reply(c, m):
        if m[0]["content"].startswith("Task type: plan."):
            return json.dumps(plan)
        lines = m[-1]["content"].split("TICKETS:\n", 1)[1].split("\n")
        return json.dumps({"rows": [{"ticket_id": x.split(" |")[0], "is_refund": False} for x in lines if x]})

    client = ScriptedClient(reply, sleep_s=lambda c: 0.12)
    c5 = cand("C5", max_concurrency=3)
    gw = Gateway({"C5": c5}, {"C5": client}, Router("fixed", fixed="C5"), Tracer("w"), RealClock())
    q = build_questions(tickets_8k, "A", 1, 1, "8k")[0]
    ctx = RunContext(gw, "w", {"8k": tickets_8k}, planner="C5")
    t0 = time.perf_counter()
    rec = (await run_pipeline("P1", [q], ctx))[0]
    wall = (time.perf_counter() - t0) * 1000
    assert abs(critical_path_ms(rec.spans) - wall) / wall < 0.05
    assert abs(rec.latency_ms - rec.wall_ms) / rec.wall_ms < 0.05


def test_run_files_round_trip_byte_identically(tmp_path):
    r = RunRecord(run_id="x", pipeline="P1", question_id="q", cls="A", size="8k", correct=True)
    write_run(tmp_path / "a.jsonl.gz", {"run_id": "x"}, [r])
    write_run(tmp_path / "b.jsonl.gz", {"run_id": "x"}, [r])
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    meta, recs = read_run(tmp_path / "a.jsonl.gz")
    assert meta == {"run_id": "x"} and recs == [r]


async def test_sim_runs_are_deterministic(world_8k, tickets_8k):
    qs = questions(world_8k)
    a = await run_pipeline("P2", qs, ctx_for("P2", world_8k, tickets_8k, "rules"))
    b = await run_pipeline("P2", qs, ctx_for("P2", world_8k, tickets_8k, "rules"))
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


async def test_offline_replay_follows_recorded_live_routing(tmp_path):
    """Both candidates have a cached answer for the same request; the replay must use the one the live run used."""
    from mosaic.config import ReplayOnlyClient
    from mosaic.providers.base import Completion
    from mosaic.providers.cache import ResponseCache, cache_key
    from mosaic.types import CallSpec
    cache = ResponseCache(tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "x"}]
    a, b = cand("C4", price_in=0.01, price_out=0.01), cand("C5", price_in=1.0, price_out=1.0)
    for c, text in ((a, '{"v": "from C4"}'), (b, '{"v": "from C5"}')):
        params = {"max_tokens": 10, "temperature": 0.0, "reasoning_effort": c.reasoning_effort, "json": True}
        cache.put(cache_key(c.provider, c.model, msgs, params), c.provider, c.model,
                  Completion(text=text, tokens_in=5, tokens_out=5, ttft_ms=1, total_ms=2))
    replay = ReplayOnlyClient()
    gw = Gateway({"C4": a, "C5": b}, {"C4": replay, "C5": replay}, Router("rules"), Tracer("r"), VirtualClock(),
                 cache=cache)
    spec = CallSpec(role="mapper", tokens_in_est=5, tokens_out_est=5, on_critical_path=False)
    first = await gw.call(spec, msgs, 10, 0.0, "q")
    assert first.text == '{"v": "from C4"}'  # the router alone picks the cheaper candidate
    gw.replay_hints = {("q", first.span.request_id): ["C5"]}
    again = await gw.call(spec, msgs, 10, 0.0, "q")
    assert again.text == '{"v": "from C5"}' and again.span.candidate_id == "C5"
