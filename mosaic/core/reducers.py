"""Deterministic reducers: code does arithmetic and joins (PRD 6)."""
from __future__ import annotations

from typing import Any

from mosaic.data.oracle import pair_customers


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


def matches(row: dict, where: dict) -> bool:
    for k, v in where.items():
        if _norm(row.get(k)) != _norm(v):
            return False
    return True


def _amount(v: Any) -> float:
    if v is None or isinstance(v, bool):
        return 0.0
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def reduce_rows(rows: list[dict], reduce: str, args: dict) -> Any:
    where = args.get("where", {})
    if reduce == "count":
        return sum(1 for r in rows if matches(r, where))
    if reduce == "sum":
        return round(sum(_amount(r.get(args["field"])) for r in rows if matches(r, where)), 2)
    if reduce == "set_union":
        vals = sorted({r.get(args["field"]) for r in rows if matches(r, where) and r.get(args["field"])})
        return vals
    if reduce == "pair_join":
        norm = [{**r, args["field"]: _norm(r.get(args["field"]))} for r in rows]
        ids = pair_customers(norm, args["field"], _norm(args["a"]), _norm(args["b"]), float(args["window_days"]))
        return len(ids) if args.get("output") == "count" else ids
    raise ValueError(f"no deterministic reducer for {reduce!r}")


def to_answer(key: str, value: Any) -> dict:
    if key == "ticket_id":
        value = value[0] if isinstance(value, list) and value else (value or None)
    return {key: value}
