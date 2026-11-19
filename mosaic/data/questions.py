"""Question builder (PRD 5, 9.2). Each question carries its template and parameters so the
oracle, the reference planner and the simulator can all interpret it exactly."""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any

from mosaic.data.gen import CATEGORIES, PRODUCTS
from mosaic.data.oracle import oracle
from mosaic.types import Question, Ticket

MONTHS = ["January", "February", "March", "April", "May", "June"]
CAT_NAME = {"billing": "billing", "shipping": "shipping", "defect": "product defect",
            "account": "login and profile access", "other": "general enquiry"}
WINDOWS = [2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60]
ANSWER_TYPE = {"L_refund": "amount", "L_customer": "customer_ids", "L_ticket": "ticket_id",
               "A_refund_count": "count", "A_refund_sum": "amount", "A_sentiment_count": "count",
               "A_category_count": "count", "P_pair_count": "count", "P_pair_ids": "customer_ids",
               "P_repeat_negative": "count"}


def _month_options() -> list[list[int] | None]:
    opts: list[list[int] | None] = [None] + [[m] for m in range(1, 7)]
    opts += [list(range(a, b + 1)) for a, b in itertools.combinations(range(1, 7), 2)]
    return opts


def _when(months: list[int] | None) -> str:
    if not months:
        return ""
    if len(months) == 1:
        return f" in {MONTHS[months[0] - 1]}"
    return f" between {MONTHS[months[0] - 1]} and {MONTHS[months[-1] - 1]}"


def render(template: str, p: dict[str, Any]) -> str:
    w = _when(p.get("months"))
    if template == "L_refund":
        return f"What refund amount did ticket {p['ticket_id']} receive?"
    if template == "L_customer":
        return f"Which customer filed ticket {p['ticket_id']}?"
    if template == "L_ticket":
        return f"Which ticket did customer {p['customer_id']} file on {p['date']}?"
    if template == "A_refund_count":
        return f"How many tickets about {p['product']} asked for a refund{w}?"
    if template == "A_refund_sum":
        return f"What was the total amount refunded for {CAT_NAME[p['category']]} issues{w}?"
    if template == "A_sentiment_count":
        return f"How many {p['product']} tickets{w} were written in a {p['sentiment']} tone?"
    if template == "A_category_count":
        return f"How many tickets{w} were about {CAT_NAME[p['category']]} problems?"
    if template == "P_pair_count":
        return (f"How many customers filed both a {CAT_NAME[p['a']]} complaint and a {CAT_NAME[p['b']]} "
                f"complaint within {p['window_days']} days of each other?")
    if template == "P_pair_ids":
        return (f"Which customers filed both a {CAT_NAME[p['a']]} complaint and a {CAT_NAME[p['b']]} "
                f"complaint within {p['window_days']} days of each other?")
    if template == "P_repeat_negative":
        return f"How many customers wrote two or more negative-tone tickets within {p['window_days']} days?"
    raise ValueError(template)


def param_space(cls: str, tickets: list[Ticket]) -> list[tuple[str, dict]]:
    """Every (template, params) for a class. Lookups depend on the corpus; the rest do not."""
    out: list[tuple[str, dict]] = []
    if cls == "L":
        for t in tickets:
            if t.refund_amount:
                out.append(("L_refund", {"ticket_id": t.ticket_id}))
            out.append(("L_customer", {"ticket_id": t.ticket_id}))
            if sum(1 for u in tickets if u.customer_id == t.customer_id and u.ts[:10] == t.ts[:10]) == 1:
                out.append(("L_ticket", {"customer_id": t.customer_id, "date": t.ts[:10]}))
    elif cls == "A":
        for m in _month_options():
            for prod in PRODUCTS:
                out.append(("A_refund_count", {"product": prod, "months": m}))
                for s in ("negative", "positive"):
                    out.append(("A_sentiment_count", {"product": prod, "sentiment": s, "months": m}))
            for c in CATEGORIES:
                out.append(("A_refund_sum", {"category": c, "months": m}))
                out.append(("A_category_count", {"category": c, "months": m}))
    else:
        pairs = list(itertools.combinations(CATEGORIES, 2)) + [(c, c) for c in CATEGORIES]
        for (a, b), w in itertools.product(pairs, WINDOWS):
            out.append(("P_pair_count", {"a": a, "b": b, "window_days": w}))
            out.append(("P_pair_ids", {"a": a, "b": b, "window_days": w}))
        for w in WINDOWS:
            out.append(("P_repeat_negative", {"a": "negative", "b": "negative", "window_days": w}))
    return out


def text_split(text: str) -> str:
    """Deterministic 70/30 split of question texts into evaluation and profiling pools."""
    return "profiling" if int(hashlib.sha256(text.encode()).hexdigest(), 16) % 10 < 3 else "eval"


def _is_zero(v: Any) -> bool:
    return v in (0, 0.0, None, [], "")


def build_questions(tickets: list[Ticket], cls: str, n: int, seed: int, size: str,
                    split: str = "eval", max_zero_share: float = 0.10) -> list[Question]:
    rng = random.Random(f"mosaic-q-{seed}-{cls}-{size}-{split}")
    space = [(tpl, p) for tpl, p in param_space(cls, tickets) if text_split(render(tpl, p)) == split]
    rng.shuffle(space)
    if cls == "L":  # balance lookup templates
        space.sort(key=lambda x: (rng.random() + {"L_refund": 0, "L_customer": 0.3, "L_ticket": 0.15}[x[0]]))
    chosen, zeros, seen = [], [], set()
    for tpl, p in space:
        text = render(tpl, p)
        if text in seen:
            continue
        truth = oracle({"template": tpl, "params": p}, tickets)
        if cls == "L" and truth is None:
            continue
        (zeros if _is_zero(truth) else chosen).append((tpl, p, text, truth))
        seen.add(text)
        if len(chosen) >= n:
            break
    n_zero = min(len(zeros), int(n * max_zero_share), max(0, n - len(chosen)))
    picked = chosen[: n - n_zero] + zeros[:n_zero]
    rng.shuffle(picked)
    prefix = "q" if split == "eval" else "pq"
    return [
        Question(question_id=f"{prefix}-{cls.lower()}-{size}-{i:03d}", cls=cls, size=size, text=text,
                 answer_type=ANSWER_TYPE[tpl], truth=truth, seed=seed, template=tpl, params=p)
        for i, (tpl, p, text, truth) in enumerate(picked)
    ]
