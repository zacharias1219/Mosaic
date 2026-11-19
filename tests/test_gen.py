"""9.1 Log generator."""
import re
from collections import Counter

import numpy as np
from scipy import stats

from mosaic.data.gen import CATEGORIES, ZIPF_S, corpus_text, dumps, generate
from mosaic.tokens import count_tokens


def test_same_seed_byte_identical_and_different_seeds_differ():
    assert dumps(generate(8000, 7)) == dumps(generate(8000, 7))
    assert dumps(generate(8000, 7)) != dumps(generate(8000, 8))


def test_token_count_within_5_percent():
    for n in (8000, 32000, 128000):
        tok = count_tokens(corpus_text(generate(n, 42)))
        assert abs(tok - n) / n <= 0.05


def test_refund_tickets_and_only_they_have_amounts(tickets_32k):
    for t in tickets_32k:
        assert (t.refund_amount is not None) == bool(t.refund_amount)
    assert any(t.refund_amount for t in tickets_32k)
    # amounts are expressed in text for refund tickets
    for t in tickets_32k:
        if t.refund_amount:
            assert f"${t.refund_amount:.2f}" in t.text


def test_category_words_never_verbatim(tickets_32k):
    pat = re.compile(r"\b(" + "|".join(CATEGORIES) + r")\b", re.I)
    assert not [t.ticket_id for t in tickets_32k if pat.search(t.text)]


def test_at_least_30_percent_refunds_paraphrased(tickets_32k):
    refunds = [t for t in tickets_32k if t.refund_amount]
    para = sum(1 for t in refunds if "refund" not in t.text.lower())
    assert para / len(refunds) >= 0.30


def test_customer_frequency_follows_zipf():
    tickets = generate(512000, 3)
    counts = Counter(int(t.customer_id[2:]) for t in tickets)
    pool = max(counts)
    k = 15  # top ranks carry most of the mass; bin the tail
    w = np.array([1 / r ** ZIPF_S for r in range(1, pool + 1)])
    w /= w.sum()
    n = len(tickets)
    obs = [counts.get(r, 0) for r in range(1, k + 1)] + [sum(v for r, v in counts.items() if r > k)]
    exp = list(n * w[:k]) + [n * w[k:].sum()]
    assert stats.chisquare(obs, exp).pvalue > 0.001


def test_tickets_sorted_unique_and_spread_over_six_months(tickets_32k):
    ids = [t.ticket_id for t in tickets_32k]
    assert len(set(ids)) == len(ids)
    assert [t.ts for t in tickets_32k] == sorted(t.ts for t in tickets_32k)
    assert {t.ts[5:7] for t in tickets_32k} == {"01", "02", "03", "04", "05", "06"}
