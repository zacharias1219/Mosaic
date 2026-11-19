"""9.7 Planner and plan validator."""
import json

import pytest

from mosaic.core.plan_schema import PlanError, validate_plan
from mosaic.core.planner import make_plan, reference_plan
from mosaic.data.questions import build_questions
from tests.helpers import ScriptedClient, cand, gateway

GOOD = {"strategy": "recurse", "filter": {"keywords": ["refund"], "fields": {"product": "Mesh Pro", "months": [3]},
                                          "semantic": True}, "chunk_tokens": 3000, "map_task": "x",
        "map_schema": {"is_refund": "bool"}, "reduce": "count", "reduce_args": {"where": {"is_refund": True}}}


def test_valid_plan_round_trips():
    p = validate_plan(json.dumps(GOOD))
    assert p.answer_key() == "count" and p.filter.fields["months"] == [3]


@pytest.mark.parametrize("bad, words", [
    ("not json at all", "no valid JSON"),
    ({**GOOD, "code": "import os"}, "code"),
    ({**GOOD, "filter": {**GOOD["filter"], "region": "EU"}}, "region"),
    ({**GOOD, "filter": {**GOOD["filter"], "fields": {"colour": "red"}}}, "colour"),
    ({**GOOD, "reduce": "sum"}, "amount"),
    ({**GOOD, "map_schema": {}}, "map_schema"),
    ({**GOOD, "map_schema": {"x": "float"}}, "type"),
    ({**GOOD, "reduce_args": {"where": {"nope": 1}}}, "where"),
])
def test_invalid_plans_fail_with_readable_error(bad, words):
    with pytest.raises(PlanError) as e:
        validate_plan(bad if isinstance(bad, str) else json.dumps(bad))
    assert words in str(e.value)


async def test_planner_retries_once_with_the_error():
    replies = iter(['{"strategy": "recurse", "bogus": 1}', json.dumps(GOOD)])
    client = ScriptedClient(lambda c, m: next(replies))
    gw = gateway([cand("A")], client)
    plan, _, notes = await make_plan("How many?", "", gw, 0.0, "q1")
    assert plan.reduce == "count" and len(client.calls) == 2
    assert "invalid" in client.calls[1][1][-1]["content"] and notes


async def test_planner_gives_up_after_two_invalid():
    gw = gateway([cand("A")], ScriptedClient(lambda c, m: "nope"))
    with pytest.raises(PlanError):
        await make_plan("How many?", "", gw, 0.0, "q1")


def test_reference_plans_for_fixture_classes(tickets_8k):
    for q in build_questions(tickets_8k, "L", 30, 3, "8k"):
        p = reference_plan(q)
        assert p.strategy == "direct" and set(p.filter.fields) & {"ticket_id", "customer_id"}
    for q in build_questions(tickets_8k, "A", 60, 3, "8k"):
        p = reference_plan(q)
        if q.template == "A_refund_sum":
            assert p.reduce == "sum"
        assert p.reduce != "llm" and p.filter.semantic
        assert p.answer_key() == q.answer_type
    for q in build_questions(tickets_8k, "P", 30, 3, "8k"):
        assert reference_plan(q).reduce == "pair_join" and reference_plan(q).answer_key() == q.answer_type


async def test_sim_planner_on_fixtures(world_8k):
    """The planner role end to end (simulated large model): lookups go direct, sums use reduce=sum,
    and paraphrase-heavy questions get semantic=true in at least 80% of fixtures."""
    from mosaic.providers.sim import SimClient
    sim = SimClient(world_8k)
    big = cand("C3", sim={"tier": "large", "plan_err": 0.04, "malformed": 0.0})
    gw = gateway([big], sim)
    qs = list(world_8k.questions.values())
    direct = semantic = n_para = 0
    lookups = [q for q in qs if q.cls == "L"]
    for q in qs:
        plan, _, _ = await make_plan(q.text, "", gw, 0.0, q.question_id)
        if q.cls == "L":
            direct += plan.strategy == "direct" or "ticket_id" in plan.filter.fields
        if q.template == "A_refund_sum":
            assert plan.reduce == "sum"
        if q.template in ("A_refund_count", "A_refund_sum"):
            n_para += 1
            semantic += plan.filter.semantic
    assert direct / len(lookups) >= 0.9
    assert semantic / n_para >= 0.8
