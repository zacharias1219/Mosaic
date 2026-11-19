"""Compare a model answer with the oracle (PRD 5, 9.3)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from mosaic.core.plan_schema import PlanError, extract_json


class Score(BaseModel):
    correct: bool
    malformed: bool = False
    extra_prose: bool = False
    jaccard: Optional[float] = None
    value: Any = None


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def score_value(obj: Any, answer_type: str, truth: Any) -> Score:
    if not isinstance(obj, dict) or answer_type not in obj:
        return Score(correct=False, malformed=True)
    v = obj[answer_type]
    if answer_type == "count":
        n = _num(v)
        ok = n is not None and truth is not None and float(n).is_integer() and int(n) == int(truth)
        return Score(correct=ok, malformed=n is None, value=v)
    if answer_type == "amount":
        n = _num(v)
        ok = n is not None and truth is not None and abs(n - float(truth)) <= 0.01 + 1e-9
        return Score(correct=ok, malformed=n is None and v is not None, value=v)
    if answer_type == "ticket_id":
        ok = isinstance(v, str) and v.strip().upper() == str(truth).upper()
        return Score(correct=ok, malformed=not isinstance(v, (str, type(None))), value=v)
    if answer_type == "customer_ids":
        if not isinstance(v, list):
            return Score(correct=False, malformed=True, value=v)
        a, b = {str(x).strip().upper() for x in v}, {str(x).upper() for x in (truth or [])}
        jac = len(a & b) / len(a | b) if (a | b) else 1.0
        return Score(correct=a == b, jaccard=jac, value=sorted(a))
    raise ValueError(answer_type)


def score(pred_json: Optional[str | dict], truth: Any, answer_type: str) -> Score:
    if pred_json is None:
        return Score(correct=False, malformed=True)
    if isinstance(pred_json, dict):
        return score_value(pred_json, answer_type, truth)
    try:
        obj, extra = extract_json(pred_json)
    except PlanError:
        return Score(correct=False, malformed=True)
    s = score_value(obj, answer_type, truth)
    s.extra_prose = extra
    return s
