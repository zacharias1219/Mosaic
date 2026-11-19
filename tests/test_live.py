"""Opt-in live suite: `pytest -m live`. Calls real providers through the response cache, so a
second run costs nothing. Requires network access to Groq (key or proxy) and data from `mosaic gen`."""
import pytest

from mosaic.config import ROOT, load_candidates, load_config
from mosaic.core.planner import make_plan
from mosaic.data.store import load_corpus, load_questions
from mosaic.pipelines import sample_lines
from mosaic.session import Session
from mosaic.trace import Tracer

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def planner_gw():
    s = Session(load_config(), load_candidates())
    if not (s.data_dir / "questions-eval.jsonl").exists():
        pytest.skip("run `mosaic gen` first")
    return s, s.gateway(s.router("P1", pool=["C3"]), {"default": "sim", "C3": "live"}, Tracer("live-test"), live=True)


async def _plans(s, gw, n=4):
    allq = load_questions(s.data_dir, "eval")
    qs = [q for c in "LAP" for q in [x for x in allq if x.size == "8k" and x.cls == c][:n]]
    lines = [t.line() for t in load_corpus(s.data_dir, "eval", "8k")]
    out = []
    for q in qs:
        plan, _, _ = await make_plan(q.text, sample_lines(lines, 800), gw, gw.clock.now(), q.question_id, pin="C3")
        out.append((q, plan))
    return out


async def test_live_planner_lookups_go_direct(planner_gw):
    s, gw = planner_gw
    for q, p in await _plans(s, gw):
        if q.cls == "L":
            assert p.strategy == "direct" or "ticket_id" in p.filter.fields


async def test_live_planner_sums_use_code(planner_gw):
    s, gw = planner_gw
    for q, p in await _plans(s, gw):
        if q.template == "A_refund_sum":
            assert p.reduce == "sum"
        if q.cls in "AP":
            assert p.strategy == "recurse" and p.reduce != "llm"


async def test_live_planner_semantic_for_paraphrase_heavy(planner_gw):
    s, gw = planner_gw
    plans = [p for q, p in await _plans(s, gw) if q.cls in "AP"]
    assert sum(p.filter.semantic for p in plans) / len(plans) >= 0.8


async def test_live_groq_call_respects_cap():
    from mosaic.providers.groq import GroqClient
    c = load_candidates()["C1"]
    client = GroqClient(load_config()["providers"]["groq"]["base_url"])
    comp = await client.complete(c, [{"role": "user", "content": 'Return JSON {"ok": true}'}], max_tokens=200)
    assert comp.tokens_in > 0 and max(client.sent_estimates) <= c.max_call_tokens
