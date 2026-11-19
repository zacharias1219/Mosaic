"""9.2 Question builder and oracle."""
import datetime as dt
import json
from pathlib import Path

from mosaic.data.gen import generate
from mosaic.data.oracle import brute_force_pairs, oracle
from mosaic.data.questions import build_questions
from mosaic.types import Ticket

FIX = Path(__file__).parent / "fixtures"


def test_lookups_target_existing_tickets(tickets_8k):
    ids = {t.ticket_id for t in tickets_8k}
    custs = {t.customer_id for t in tickets_8k}
    for q in build_questions(tickets_8k, "L", 40, 1, "8k"):
        if "ticket_id" in q.params:
            assert q.params["ticket_id"] in ids
        else:
            assert q.params["customer_id"] in custs and q.truth in ids


def test_aggregate_answers_match_hand_checked_fixture():
    data = json.loads((FIX / "agg_fixture.json").read_text())
    tickets = [Ticket(**t) for t in data["tickets"]]
    assert len(tickets) == 20
    for case in data["cases"]:
        assert oracle(case, tickets) == case["expected"], case


def test_pairwise_matches_brute_force_on_small_corpora():
    for seed in range(5):
        tickets = generate(8000, seed)
        for a, b, w in [("shipping", "defect", 7), ("billing", "billing", 14), ("account", "other", 30)]:
            q = {"template": "P_pair_ids", "params": {"a": a, "b": b, "window_days": w}}
            assert oracle(q, tickets) == brute_force_pairs(tickets, a, b, w)


def test_zero_answers_at_most_10_percent(tickets_8k):
    for cls in "LAP":
        qs = build_questions(tickets_8k, cls, 60, 42, "8k")
        zeros = sum(1 for q in qs if q.truth in (0, 0.0, None, []))
        assert zeros <= 0.10 * len(qs)


def test_eval_and_profiling_share_no_question_text(tickets_8k, tickets_32k):
    ev = {q.text for t, s in ((tickets_8k, "8k"), (tickets_32k, "32k")) for c in "LAP"
          for q in build_questions(t, c, 60, 42, s, split="eval")}
    pr = {q.text for t, s in ((tickets_8k, "8k"), (tickets_32k, "32k")) for c in "LAP"
          for q in build_questions(t, c, 25, 43, s, split="profiling")}
    assert ev and pr and not (ev & pr)


def test_question_ids_unique_and_truth_matches_oracle(tickets_8k):
    qs = [q for c in "LAP" for q in build_questions(tickets_8k, c, 30, 5, "8k")]
    assert len({q.question_id for q in qs}) == len(qs)
    for q in qs:
        assert oracle(q, tickets_8k) == q.truth
