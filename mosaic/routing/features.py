"""Call features the predictor sees. All are computable in code before the call is made."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mosaic.tokens import count_tokens

if TYPE_CHECKING:
    from mosaic.core.executor import Record
    from mosaic.core.plan_schema import Plan

CUES = ("refund", "money back", "reimburse")


def map_features(recs: list["Record"], plan: "Plan") -> dict[str, float]:
    n = max(1, len(recs))
    toks = sum(count_tokens(r.line) for r in recs)
    types = set(plan.map_schema.values())
    cue = sum(1 for r in recs if any(c in r.text.lower() for c in CUES)) / n
    money = sum(1 for r in recs if "$" in r.text) / n
    return {
        "log_tokens": math.log2(1 + toks), "n_tickets": float(len(recs)), "n_fields": float(len(plan.map_schema)),
        "has_bool": float("bool" in types), "has_amount": float("amount" in types),
        "has_category": float("category" in types), "has_sentiment": float("sentiment" in types),
        "semantic": float(plan.filter.semantic), "pair": float(plan.reduce == "pair_join"),
        "cue_share": cue, "money_share": money, "uncued_money": max(0.0, money - cue),
    }


def direct_features(lines: list[str]) -> dict[str, float]:
    toks = sum(count_tokens(x) for x in lines)
    return {"log_tokens": math.log2(1 + toks), "n_tickets": float(len(lines))}


FEATURES = {
    "mapper": ["log_tokens", "n_tickets", "n_fields", "has_bool", "has_amount", "has_category", "has_sentiment",
               "semantic", "pair", "cue_share", "money_share", "uncued_money"],
    "direct": ["log_tokens", "n_tickets"],
    "planner": [],
    "aggregator": ["log_tokens"],
}


def vector(role: str, feats: dict[str, float]) -> list[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURES[role]]
