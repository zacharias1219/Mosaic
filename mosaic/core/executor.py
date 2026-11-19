"""Executor: filter and chunk in code, map with routed models, reduce in code (PRD 9.8)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from mosaic.core.plan_schema import Plan, PlanError, extract_json
from mosaic.core.prompts import agg_messages, direct_messages, map_messages
from mosaic.core.reducers import reduce_rows, to_answer
from mosaic.routing.features import direct_features, map_features
from mosaic.tokens import TOKEN_MARGIN, count_tokens
from mosaic.types import CallSpec

if TYPE_CHECKING:
    from mosaic.gateway import Gateway

Labeler = Callable[[list[str], list[dict], Plan], bool]


@dataclass
class Record:
    ticket_id: str
    ts: str
    customer_id: str
    product: str
    text: str
    line: str


def parse_log(lines: list[str]) -> list[Record]:
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|", 4)]
        if len(parts) == 5:
            out.append(Record(*parts, line=ln))
    return out


def apply_filter(recs: list[Record], plan: Plan) -> tuple[list[Record], dict[str, int]]:
    f = plan.filter.fields
    out = []
    for r in recs:
        if "product" in f and r.product.lower() != str(f["product"]).lower():
            continue
        if "months" in f and int(r.ts[5:7]) not in f["months"]:
            continue
        if "customer_id" in f and r.customer_id != f["customer_id"]:
            continue
        if "ticket_id" in f and r.ticket_id != f["ticket_id"]:
            continue
        if "date" in f and r.ts[:10] != f["date"]:
            continue
        out.append(r)
    kws = [k.lower() for k in plan.filter.keywords]
    hits = {k: sum(1 for r in out if k in r.text.lower()) for k in kws}
    if kws and not plan.filter.semantic:  # keywords are trusted only when semantic=false
        out = [r for r in out if any(k in r.text.lower() for k in kws)]
    return out, hits


def chunk(recs: list[Record], chunk_tokens: int) -> list[list[Record]]:
    """Greedy packing that never splits a ticket; an oversized ticket gets its own chunk."""
    chunks: list[list[Record]] = []
    cur: list[Record] = []
    size = 0
    for r in recs:
        t = count_tokens(r.line) + 1
        if cur and size + t > chunk_tokens:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += t
    if cur:
        chunks.append(cur)
    return chunks


def _coerce(v: Any, typ: str) -> Any:
    if typ == "bool":
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
    if typ == "amount":
        if v in (None, "", "null", False):
            return None
        try:
            return float(str(v).replace("$", "").replace(",", ""))
        except ValueError:
            return None
    if typ in ("category", "sentiment") and isinstance(v, str):
        return v.strip().lower()
    return v


def parse_rows(text: str, recs: list[Record], schema: dict[str, str]) -> tuple[list[dict], bool]:
    """Returns (rows joined with header metadata, malformed?)."""
    try:
        obj, _ = extract_json(text)
    except PlanError:
        return [], True
    rows = obj.get("rows") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        return [], True
    meta = {r.ticket_id: r for r in recs}
    out, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("ticket_id") not in meta or row["ticket_id"] in seen:
            continue
        seen.add(row["ticket_id"])
        r = meta[row["ticket_id"]]
        facts = {k: _coerce(row.get(k), t) for k, t in schema.items()}
        out.append({"ticket_id": r.ticket_id, "ts": r.ts, "customer_id": r.customer_id, "product": r.product, **facts})
    return out, False


@dataclass
class ExecResult:
    answer: Optional[dict]
    end: float
    notes: list[str] = field(default_factory=list)
    malformed: bool = False
    truncated: bool = False
    filter_hits: dict[str, int] = field(default_factory=dict)
    n_after_filter: int = 0
    n_chunks: int = 0
    depth_cap_hit: bool = False


class Executor:
    def __init__(self, gateway: "Gateway", max_depth: int = 2, labeler: Optional[Labeler] = None):
        self.gw = gateway
        self.max_depth = max_depth
        self.labeler = labeler

    async def run(self, plan: Plan, lines: list[str], question: str, answer_key: str, at: float, qid: str,
                  parent: Optional[str]) -> ExecResult:
        recs, hits = apply_filter(parse_log(lines), plan)
        span = self.gw.tracer.add(qid, "filter", parent, start=at, end=at,
                                  note=f"{len(lines)} -> {len(recs)} tickets; keyword hits {hits}")
        res = ExecResult(None, at, filter_hits=hits, n_after_filter=len(recs))
        res.notes.append(span.note)
        if plan.strategy == "direct":
            return await self.direct(recs, question, answer_key, at, qid, parent, res)
        chunks = chunk(recs, plan.chunk_tokens)
        res.n_chunks = len(chunks)
        outs = await asyncio.gather(*[self.map_chunk(c, plan, at, qid, parent, 0, res) for c in chunks])
        rows = [r for rs, _ in outs for r in rs]
        end = max([e for _, e in outs], default=at)
        if plan.reduce == "llm":
            return await self.aggregate(rows, question, answer_key, end, qid, parent, res)
        value = reduce_rows(rows, plan.reduce, plan.reduce_args)
        self.gw.tracer.add(qid, "reduce", parent, start=end, end=end, note=f"code ({plan.reduce}) over {len(rows)} rows")
        res.answer, res.end = to_answer(plan.answer_key(), value), end
        return res

    async def map_chunk(self, recs: list[Record], plan: Plan, at: float, qid: str, parent: Optional[str],
                        depth: int, res: ExecResult) -> tuple[list[dict], float]:
        lines = [r.line for r in recs]
        msgs = map_messages(plan.map_task, plan.map_schema, lines)
        out_est = len(recs) * (10 + 7 * len(plan.map_schema)) + 30
        spec = CallSpec(role="mapper", tokens_in_est=sum(count_tokens(m["content"]) for m in msgs),
                        tokens_out_est=out_est, on_critical_path=False, features=map_features(recs, plan))
        exclude: set[str] = set()
        end = at
        for _attempt in range(3):
            call = await self.gw.call(spec, msgs, max_tokens=out_est * 2 + 200, at=end, qid=qid, parent_id=parent,
                                      exclude=exclude)
            end = call.end
            if call.too_large and len(recs) > 1:
                if depth >= self.max_depth:
                    res.depth_cap_hit = True
                    res.notes.append(f"recursion depth cap {self.max_depth} hit on a {len(recs)}-ticket chunk")
                    return [], end
                half = len(recs) // 2
                parts = await asyncio.gather(self.map_chunk(recs[:half], plan, at, qid, parent, depth + 1, res),
                                             self.map_chunk(recs[half:], plan, at, qid, parent, depth + 1, res))
                return parts[0][0] + parts[1][0], max(parts[0][1], parts[1][1])
            if call.text is None:
                res.notes.append(f"map chunk of {len(recs)} failed on every candidate")
                return [], end
            rows, bad = parse_rows(call.text, recs, plan.map_schema)
            if self.labeler and call.span is not None:
                call.span.correct = (not bad) and self.labeler([r.ticket_id for r in recs], rows, plan)
            if not bad:
                return rows, end
            call.span.note = "malformed map output; falling back"
            exclude.add(call.span.candidate_id)
        res.notes.append("map output malformed on all attempts")
        return [], end

    async def direct(self, recs: list[Record], question: str, answer_key: str, at: float, qid: str,
                     parent: Optional[str], res: ExecResult) -> ExecResult:
        lines = [r.line for r in recs]
        window = self.gw.max_window("direct")
        budget = int(window / TOKEN_MARGIN) - 1500
        kept, used = [], 0
        for ln in lines:
            t = count_tokens(ln) + 1
            if used + t > budget:
                res.truncated = True
                break
            kept.append(ln)
            used += t
        if res.truncated:
            res.notes.append(f"direct context truncated: {len(kept)}/{len(lines)} tickets fit the {window}-token window")
        msgs = direct_messages(question, answer_key, kept)
        spec = CallSpec(role="direct", tokens_in_est=sum(count_tokens(m["content"]) for m in msgs), tokens_out_est=60,
                        on_critical_path=True, features=direct_features(kept))
        call = await self.gw.call(spec, msgs, max_tokens=800, at=at, qid=qid, parent_id=parent)
        res.end = call.end
        return self._final(call.text, answer_key, res)

    async def aggregate(self, rows: list[dict], question: str, answer_key: str, at: float, qid: str,
                        parent: Optional[str], res: ExecResult) -> ExecResult:
        msgs = agg_messages(question, answer_key, rows)
        spec = CallSpec(role="aggregator", tokens_in_est=sum(count_tokens(m["content"]) for m in msgs),
                        tokens_out_est=80, on_critical_path=True, features={"log_tokens": len(rows)})
        call = await self.gw.call(spec, msgs, max_tokens=800, at=at, qid=qid, parent_id=parent)
        res.end = call.end
        return self._final(call.text, answer_key, res)

    @staticmethod
    def _final(text: Optional[str], answer_key: str, res: ExecResult) -> ExecResult:
        if text is None:
            res.malformed = True
            res.notes.append("final call failed")
            return res
        try:
            obj, _ = extract_json(text)
            res.answer = obj if isinstance(obj, dict) else None
            res.malformed = res.answer is None
        except PlanError:
            res.malformed = True
        return res
