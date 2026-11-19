"""9.3 Scorer."""
from mosaic.eval.scorer import score


def test_exact_integer_and_off_by_one():
    assert score('{"count": 23}', 23, "count").correct
    assert not score('{"count": 22}', 23, "count").correct
    assert not score('{"count": 23.5}', 23, "count").correct


def test_amount_within_one_cent():
    assert score('{"amount": 302.495}', 302.49, "amount").correct
    assert score('{"amount": "$302.49"}', 302.49, "amount").correct
    assert not score('{"amount": 302.51}', 302.49, "amount").correct


def test_id_sets_order_insensitive_with_jaccard():
    s = score('{"customer_ids": ["C-0002", "C-0001"]}', ["C-0001", "C-0002"], "customer_ids")
    assert s.correct and s.jaccard == 1.0
    s = score('{"customer_ids": ["C-0001"]}', ["C-0001", "C-0002"], "customer_ids")
    assert not s.correct and s.jaccard == 0.5


def test_malformed_json_flagged():
    for bad in ("I think 23", '{"count": ', "", None, '{"amount": 3}'):
        s = score(bad, 23, "count")
        assert not s.correct and s.malformed


def test_extra_prose_tolerated_and_counted():
    s = score('Sure! Here is the answer:\n```json\n{"count": 23}\n```\nHope that helps.', 23, "count")
    assert s.correct and s.extra_prose
    assert not score('{"count": 23}', 23, "count").extra_prose


def test_ticket_id_case_insensitive():
    assert score('{"ticket_id": "t-04821"}', "T-04821", "ticket_id").correct
