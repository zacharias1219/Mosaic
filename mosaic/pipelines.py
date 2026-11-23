"""The four systems under test (PRD 9.11, 10), all emitting the same RunRecord schema.

P0 direct: the baseline candidate reads the whole corpus (truncated to its window).
P1 recursive, fixed: shared plan; every executor call on the baseline candidate (R0).
P2 recursive, rules: shared plan; rule-table router (R1).
P3 recursive, learned: shared plan; calibrated learned router (R2).
P0f filtered direct: shared plan's filter applied in code, then ONE direct call over what is left (no
    splitting). It separates the gain from filtering in code (P0 -> P0f) from the gain from splitting the
    work into per-chunk extraction and reducing in code (P0f -> P1).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mosaic.core.executor import Executor
from mosaic.core.plan_schema import Plan, PlanError
from mosaic.core.reducers import matches
from mosaic.core.planner import make_plan, reference_plan
from mosaic.eval.scorer import score
from mosaic.gateway import Gateway
from mosaic.providers.sim import field_kind, truth_of
from mosaic.tokens import count_tokens
from mosaic.trace import critical_path_ms
from mosaic.types import Question, RunRecord, Ticket

PIPELINES = ("P0", "P0f", "P1", "P2", "P3")


def project(plan: Plan, row: dict) -> tuple:
    """What the reducer will use from one ticket's facts. Two rows with the same projection
    lead to the same answer, so a map call is judged on projections, not raw fields: a plan
    that asks only "shipping, defect or other?" is not wrong for calling a billing ticket other."""
    a = plan.reduce_args
    if plan.reduce in ("count", "sum", "set_union"):
        hit = matches(row, a.get("where", {}))
        if plan.reduce == "count":
            return (hit,)
        v = row.get(a["field"]) if hit else None
        return (hit, round(float(v), 2) if plan.reduce == "sum" and isinstance(v, (int, float)) else v)
    if plan.reduce == "pair_join":
        v = row.get(a["field"])
        v = v.strip().lower() if isinstance(v, str) else v
        return (v if v in (str(a["a"]).lower(), str(a["b"]).lower()) else None,)
    return tuple(row.get(k) for k in sorted(plan.map_schema))


def make_labeler(tickets: dict[str, Ticket]) -> Callable[[list[str], list[dict], Plan], bool]:
    """A map call is correct when every ticket's facts, as the reducer will use them, match the
    ground truth (a dropped ticket counts as having no facts)."""
    def label(ids: list[str], rows: list[dict], plan: Plan) -> bool:
        by = {r["ticket_id"]: r for r in rows}
        for tid in ids:
            t = tickets.get(tid)
            if t is None:
                return False
            meta = {"ticket_id": t.ticket_id, "ts": t.ts, "customer_id": t.customer_id, "product": t.product}
            truth = {**meta, **{n: truth_of(t, field_kind(n, ty)) for n, ty in plan.map_schema.items()}}
            got = by.get(tid, meta)
            if project(plan, got) != project(plan, truth):
                return False
        return True
    return label


def sample_lines(lines: list[str], tokens: int) -> str:
    out, used = [], 0
    for ln in lines:
        used += count_tokens(ln) + 1
        if used > tokens:
            break
        out.append(ln)
    return "\n".join(out)


@dataclass
class RunContext:
    gateway: Gateway
    run_id: str
    corpora: dict[str, list[Ticket]]
    planner: Optional[str] = None           # pinned planner candidate (shared plans)
    arrival_gap_ms: float = 60_000
    sample_tokens: int = 800
    max_depth: int = 2
    use_reference_plans: bool = False       # profiling holds the plan fixed
    progress: Optional[Callable[[int, int, RunRecord], None]] = None
    lines: dict[str, list[str]] = field(default_factory=dict)
    truth_by_size: dict[str, dict[str, Ticket]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for size, ts in self.corpora.items():
            self.lines[size] = [t.line() for t in ts]
            self.truth_by_size[size] = {t.ticket_id: t for t in ts}


async def run_question(name: str, q: Question, ctx: RunContext, at: float) -> RunRecord:
    gw, tr = ctx.gateway, ctx.gateway.tracer
    wall0 = time.perf_counter()
    lines = ctx.lines[q.size]
    root = tr.add(q.question_id, "question", None, start=at, end=at, note=q.text)
    ex = Executor(gw, max_depth=ctx.max_depth, labeler=make_labeler(ctx.truth_by_size[q.size]))
    notes: list[str] = []
    plan_d = None
    if name == "P0":
        from mosaic.core.executor import ExecResult, parse_log
        res = await ex.direct(parse_log(lines), q.text, q.answer_type, at, q.question_id, root.span_id,
                              ExecResult(None, at))
    else:
        if ctx.use_reference_plans:
            plan, t = reference_plan(q), at
        else:
            try:
                plan, t, pn = await make_plan(q.text, sample_lines(lines, ctx.sample_tokens), gw, at, q.question_id,
                                              root.span_id, pin=ctx.planner)
                notes += pn
            except PlanError as e:
                notes.append(f"planner failed twice ({e}); using fallback recursion plan")
                plan, t = reference_plan(q).model_copy(update={"strategy": "recurse"}) if q.cls != "L" else reference_plan(q), at
                t = max(t, max((s.end_ms for s in tr.spans.get(q.question_id, [])), default=at))
        if name == "P0f":  # same plan, same code filter, but one direct call instead of map + reduce
            plan = plan.model_copy(update={"strategy": "direct"})
        plan_d = plan.model_dump()
        res = await ex.run(plan, lines, q.text, q.answer_type, t, q.question_id, root.span_id)
    notes += res.notes
    spans = tr.pop(q.question_id)
    end = max([s.end_ms for s in spans] + [res.end])
    root.end_ms, root.total_ms = end, end - at
    sc = score(res.answer, q.truth, q.answer_type)
    latency = critical_path_ms(spans)
    wall = latency if gw.clock.virtual else (time.perf_counter() - wall0) * 1000  # virtual runs: virtual wall clock
    return RunRecord(
        run_id=ctx.run_id, pipeline=name, question_id=q.question_id, cls=q.cls, size=q.size,
        answer=res.answer, truth=q.truth, correct=sc.correct, malformed=sc.malformed or res.malformed,
        truncated=res.truncated, cost=sum(s.cost for s in spans), latency_ms=latency,
        wall_ms=wall, plan=plan_d, notes=notes, spans=spans)


async def run_pipeline(name: str, questions: list[Question], ctx: RunContext) -> list[RunRecord]:
    out: list[RunRecord] = []
    clock = ctx.gateway.clock
    prev_end = 0.0
    for i, q in enumerate(questions):
        if clock.virtual:  # one user: ask, get the answer, think for arrival_gap, ask again
            at = prev_end + ctx.arrival_gap_ms if i else 0.0
        else:
            at = clock.now()
        rec = await run_question(name, q, ctx, at)
        prev_end = at + rec.latency_ms
        out.append(rec)
        if ctx.progress:
            ctx.progress(i + 1, len(questions), rec)
    return out


def replay_hints(records: list[RunRecord]) -> dict[tuple[str, str], list[str]]:
    """(question, request) -> candidates that served it, in order, from a live run's spans."""
    hints: dict[tuple[str, str], list[str]] = {}
    for r in records:
        for sp in r.spans:
            if sp.request_id and sp.ok and sp.candidate_id:
                hints.setdefault((r.question_id, sp.request_id), []).append(sp.candidate_id)
    return hints


# ---- run files -----------------------------------------------------------------------------
def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "mosaic", "candidates.yaml", "mosaic.yaml"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_meta(run_id: str, pipeline: str, cfg: dict, cands: dict, data_hash: str, backends: dict, **extra) -> dict:
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"run_id": run_id, "pipeline": pipeline, "git_commit": git_commit(), "config_hash": cfg_hash,
            "data_hash": data_hash, "seed": cfg.get("seed"), "backends": backends,
            "candidates": {k: f"{c.provider}:{c.model}" for k, c in cands.items()}, **extra}


def write_run(path: Path, meta: dict, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
        f.write((json.dumps({"meta": meta}, sort_keys=True) + "\n").encode())
        for r in records:
            f.write((json.dumps(r.model_dump(), sort_keys=True, default=str) + "\n").encode())


def read_run(path: Path) -> tuple[dict, list[RunRecord]]:
    meta: dict[str, Any] = {}
    recs = []
    with gzip.open(path, "rt") as f:
        for ln in f:
            d = json.loads(ln)
            if "meta" in d:
                meta = d["meta"]
            else:
                recs.append(RunRecord(**d))
    return meta, recs
