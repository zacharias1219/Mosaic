"""Exact answers computed in code from ground-truth labels (PRD 9.2)."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from mosaic.types import Question, Ticket


def _dt(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def _in_months(t: Ticket, months: list[int] | None) -> bool:
    return not months or int(t.ts[5:7]) in months


def pair_customers(rows: list[dict], field: str, a: str, b: str, window_days: float) -> list[str]:
    """Customers with two distinct tickets, one with field==a and one with field==b, within the window.

    Shared by the oracle and the pair_join reducer so both compute the same thing.
    """
    by_cust: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get(field) in (a, b):
            by_cust[r["customer_id"]].append(r)
    hits = []
    for cust, rs in by_cust.items():
        found = False
        for i, x in enumerate(rs):
            for y in rs[i + 1:]:
                if {x.get(field), y.get(field)} >= {a, b} or (a == b and x.get(field) == y.get(field) == a):
                    if abs((_dt(x["ts"]) - _dt(y["ts"])).total_seconds()) <= window_days * 86400:
                        found = True
                        break
            if found:
                break
        if found:
            hits.append(cust)
    return sorted(hits)


def oracle(q: Question | dict, tickets: list[Ticket]) -> Any:
    template, p = (q.template, q.params) if isinstance(q, Question) else (q["template"], q["params"])
    months = p.get("months")
    if template == "L_refund":
        return next(t.refund_amount for t in tickets if t.ticket_id == p["ticket_id"])
    if template == "L_customer":
        return [next(t.customer_id for t in tickets if t.ticket_id == p["ticket_id"])]
    if template == "L_ticket":
        hits = [t.ticket_id for t in tickets if t.customer_id == p["customer_id"] and t.ts[:10] == p["date"]]
        return hits[0] if len(hits) == 1 else None
    if template == "A_refund_count":
        return sum(1 for t in tickets if t.product == p["product"] and _in_months(t, months) and t.refund_amount)
    if template == "A_refund_sum":
        return round(sum(t.refund_amount or 0 for t in tickets
                         if t.category == p["category"] and _in_months(t, months)), 2)
    if template == "A_sentiment_count":
        return sum(1 for t in tickets
                   if t.product == p["product"] and t.sentiment == p["sentiment"] and _in_months(t, months))
    if template == "A_category_count":
        return sum(1 for t in tickets if t.category == p["category"] and _in_months(t, months))
    if template in ("P_pair_count", "P_pair_ids", "P_repeat_negative"):
        field = "sentiment" if template == "P_repeat_negative" else "category"
        rows = [t.model_dump() for t in tickets]
        ids = pair_customers(rows, field, p["a"], p["b"], p["window_days"])
        return ids if template == "P_pair_ids" else len(ids)
    raise ValueError(f"unknown template {template}")


def brute_force_pairs(tickets: list[Ticket], a: str, b: str, window_days: float) -> list[str]:
    """Independent O(N^2) reference over all ticket pairs, used to test the oracle."""
    out = set()
    for i, x in enumerate(tickets):
        for j, y in enumerate(tickets):
            if i != j and x.customer_id == y.customer_id and x.category == a and y.category == b:
                if abs((_dt(x.ts) - _dt(y.ts)).total_seconds()) <= window_days * 86400:
                    out.add(x.customer_id)
    return sorted(out)
