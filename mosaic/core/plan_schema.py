"""Restricted JSON plan (PRD 9.7). The planner model never writes code; it fills this schema,
which is validated strictly before anything runs."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MAP_TYPES = {"bool", "amount", "category", "sentiment", "string"}
FILTER_FIELDS = {"product", "months", "customer_id", "ticket_id", "date"}
META_FIELDS = {"ticket_id", "ts", "customer_id", "product"}  # attached by code from the log header
REDUCERS = ("count", "sum", "set_union", "pair_join", "llm")


class PlanError(ValueError):
    pass


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    semantic: bool = True

    @field_validator("fields")
    @classmethod
    def _fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        unknown = set(v) - FILTER_FIELDS
        if unknown:
            raise ValueError(f"unknown filter field(s) {sorted(unknown)}; allowed {sorted(FILTER_FIELDS)}")
        v = {k: x for k, x in v.items() if x not in (None, "", [])}
        if "months" in v:
            m = v["months"]
            m = [m] if isinstance(m, int) else m
            if not isinstance(m, list) or not all(isinstance(i, int) and 1 <= i <= 12 for i in m):
                raise ValueError("months must be a list of month numbers 1-12")
            v["months"] = sorted(set(m))
        return v


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["direct", "recurse"]
    filter: Filter = Field(default_factory=Filter)
    chunk_tokens: int = Field(default=3000, ge=200, le=64000)
    map_task: str = ""
    map_schema: dict[str, str] = Field(default_factory=dict)
    reduce: Literal["count", "sum", "set_union", "pair_join", "llm"] = "count"
    reduce_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Plan":
        for k, t in self.map_schema.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", k) or k in META_FIELDS:
                raise ValueError(f"bad map_schema field name {k!r}")
            if t not in MAP_TYPES:
                raise ValueError(f"map_schema[{k!r}] type {t!r} not in {sorted(MAP_TYPES)}")
        if self.strategy == "direct":
            return self
        if not self.map_schema:
            raise ValueError("strategy=recurse needs a non-empty map_schema")
        known = set(self.map_schema) | META_FIELDS
        a = self.reduce_args
        where = a.get("where", {})
        if not isinstance(where, dict) or set(where) - known:
            raise ValueError(f"reduce_args.where must only use fields {sorted(known)}")
        if self.reduce == "sum" and self.map_schema.get(a.get("field", "")) != "amount":
            raise ValueError("reduce=sum needs reduce_args.field naming an 'amount' map field")
        if self.reduce == "set_union" and a.get("field") not in known:
            raise ValueError("reduce=set_union needs reduce_args.field")
        if self.reduce == "pair_join":
            if a.get("field") not in self.map_schema or "a" not in a or "b" not in a:
                raise ValueError("reduce=pair_join needs reduce_args {field, a, b, window_days, output}")
            if not isinstance(a.get("window_days"), (int, float)) or a.get("output") not in ("count", "ids"):
                raise ValueError("pair_join needs numeric window_days and output in {count, ids}")
        allowed = {"count": {"where"}, "sum": {"field", "where"}, "set_union": {"field", "where"},
                   "pair_join": {"field", "a", "b", "window_days", "output"}, "llm": set()}[self.reduce]
        if set(a) - allowed:
            raise ValueError(f"unknown reduce_args {sorted(set(a) - allowed)} for reduce={self.reduce}")
        return self

    def answer_key(self) -> str:
        if self.reduce == "sum":
            return "amount"
        if self.reduce == "pair_join":
            return "count" if self.reduce_args.get("output") == "count" else "customer_ids"
        if self.reduce == "set_union":
            return "ticket_id" if self.reduce_args.get("field") == "ticket_id" else "customer_ids"
        return "count"


def extract_json(text: str) -> tuple[Any, bool]:
    """Parse JSON, tolerating prose or code fences around one object. Returns (obj, had_extra)."""
    s = text.strip()
    try:
        return json.loads(s), False
    except (json.JSONDecodeError, ValueError):
        pass
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1]), True
                    except (json.JSONDecodeError, ValueError):
                        break
        start = s.find("{", start + 1)
    raise PlanError("no valid JSON object found")


def validate_plan(text_or_obj: str | dict) -> Plan:
    try:
        obj = extract_json(text_or_obj)[0] if isinstance(text_or_obj, str) else text_or_obj
        if not isinstance(obj, dict):
            raise PlanError("plan must be a JSON object")
        return Plan.model_validate(obj)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'plan'}: {err['msg']}" for err in e.errors())
        raise PlanError(f"invalid plan: {msgs}") from None
