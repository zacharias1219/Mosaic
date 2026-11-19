"""Planner: question -> validated Plan (PRD 9.7). The model never writes executable code."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from mosaic.core.plan_schema import Filter, Plan, PlanError, validate_plan
from mosaic.core.prompts import plan_messages
from mosaic.tokens import count_tokens
from mosaic.types import CallSpec, Question

if TYPE_CHECKING:
    from mosaic.gateway import Gateway

REFUND_KEYWORDS = ["refund", "money back", "reimburse"]
CATEGORY_KEYWORDS = {"billing": ["charge", "invoice", "billed"], "shipping": ["delivery", "courier", "tracking"],
                     "defect": ["stopped working", "broken", "rebooting"], "account": ["login", "password"],
                     "other": ["manual", "guide"]}


def reference_plan(q: Question) -> Plan:
    """What a careful engineer would write for each template. Used as the simulator's 'correct'
    plan, for planner fixtures, and as the fallback when the planner fails twice."""
    p, t = q.params, q.template
    months = {"months": p["months"]} if p.get("months") else {}
    if t.startswith("L_"):
        fields = {"ticket_id": p["ticket_id"]} if "ticket_id" in p else {"customer_id": p["customer_id"],
                                                                           "date": p["date"]}
        return Plan(strategy="direct", filter=Filter(fields=fields))
    if t == "A_refund_count":
        return Plan(strategy="recurse", filter=Filter(keywords=REFUND_KEYWORDS, fields={"product": p["product"], **months}),
                    map_task="Decide whether each ticket asks for or was promised a refund.",
                    map_schema={"is_refund": "bool"}, reduce="count", reduce_args={"where": {"is_refund": True}})
    if t == "A_refund_sum":
        return Plan(strategy="recurse", filter=Filter(keywords=REFUND_KEYWORDS, fields=months),
                    map_task="Classify each ticket's category and extract any refund amount.",
                    map_schema={"category": "category", "refund_amount": "amount"}, reduce="sum",
                    reduce_args={"field": "refund_amount", "where": {"category": p["category"]}})
    if t == "A_sentiment_count":
        return Plan(strategy="recurse", filter=Filter(fields={"product": p["product"], **months}),
                    map_task="Classify the customer's tone in each ticket.", map_schema={"sentiment": "sentiment"},
                    reduce="count", reduce_args={"where": {"sentiment": p["sentiment"]}})
    if t == "A_category_count":
        return Plan(strategy="recurse", filter=Filter(keywords=CATEGORY_KEYWORDS[p["category"]], fields=months),
                    map_task="Classify the category of each ticket.", map_schema={"category": "category"},
                    reduce="count", reduce_args={"where": {"category": p["category"]}})
    if t in ("P_pair_count", "P_pair_ids"):
        return Plan(strategy="recurse", map_task="Classify the category of each ticket.",
                    map_schema={"category": "category"}, reduce="pair_join",
                    reduce_args={"field": "category", "a": p["a"], "b": p["b"], "window_days": p["window_days"],
                                 "output": "count" if t == "P_pair_count" else "ids"})
    if t == "P_repeat_negative":
        return Plan(strategy="recurse", map_task="Classify the customer's tone in each ticket.",
                    map_schema={"sentiment": "sentiment"}, reduce="pair_join",
                    reduce_args={"field": "sentiment", "a": "negative", "b": "negative",
                                 "window_days": p["window_days"], "output": "count"})
    raise ValueError(t)


async def make_plan(question: str, sample: str, gateway: "Gateway", at: float, qid: str,
                    parent_id: Optional[str] = None, pin: Optional[str] = None) -> tuple[Plan, float, list[str]]:
    """Ask the planner model for a plan; validate; retry once with the error. Returns (plan, end_ms, notes)."""
    notes: list[str] = []
    messages = plan_messages(question, sample)
    for attempt in range(2):
        spec = CallSpec(role="planner", tokens_in_est=sum(count_tokens(m["content"]) for m in messages),
                        tokens_out_est=400, on_critical_path=True, features={})
        res = await gateway.call(spec, messages, max_tokens=1200, at=at, qid=qid, parent_id=parent_id, pin=pin)
        at = res.end
        if res.text is None:
            notes.append("planner call failed")
            break
        try:
            return validate_plan(res.text), at, notes
        except PlanError as e:
            notes.append(f"plan attempt {attempt + 1} invalid: {e}")
            messages = messages + [{"role": "assistant", "content": res.text[:2000]},
                                   {"role": "user", "content": f"That plan is invalid: {e}. Return a corrected plan JSON."}]
    raise PlanError("; ".join(notes) or "planner failed")
