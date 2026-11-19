"""`mosaic demo`: one question end to end, replayed from run records or run now (PRD 12)."""
from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

from mosaic.pipelines import read_run
from mosaic.types import RunRecord, Span


def _speed(s: str) -> float:
    if s in ("max", "0", "inf"):
        return 0.0
    return float(s.rstrip("x")) or 1.0


def _fmt_ms(ms: float) -> str:
    return f"{ms:,.0f} ms"


def render(rec: RunRecord, base: RunRecord | None, cands: dict, text: str) -> list[tuple[float, str]]:
    """Lines with the logical time (ms from question start) at which each appears."""
    root = next((s for s in rec.spans if s.role == "question"), None)
    t0 = root.start_ms if root else 0.0
    out: list[tuple[float, str]] = []
    cls = {"L": "Lookup", "A": "Aggregate", "P": "Pairwise"}[rec.cls]
    out.append((0, f"Q: {text}   size: {rec.size}   class: {cls}   ({rec.pipeline}, run {rec.run_id})"))
    plans = [s for s in rec.spans if s.role == "planner" and s.ok]
    p = rec.plan or {}
    for s in plans:
        f = p.get("filter", {})
        out.append((s.end_ms - t0, f"PLAN   {s.candidate_id} {_dev(cands, s.candidate_id):18s} {_fmt_ms(s.end_ms - s.start_ms):>10s}   "
                                   f"${s.cost:.4f}   strategy={p.get('strategy')} filter={f.get('fields')} "
                                   f"semantic={f.get('semantic')} reduce={p.get('reduce')}"))
    for s in rec.spans:
        if s.role == "filter":
            out.append((s.end_ms - t0, f"FILTER code {'':16s} {'<1 ms':>10s}             {s.note}"))
    maps = [s for s in rec.spans if s.role == "mapper" and s.ok]
    if maps:
        groups: dict[str, list[Span]] = {}
        for s in maps:
            groups.setdefault(s.candidate_id, []).append(s)
        parts = []
        for cid, ss in sorted(groups.items()):
            slow = max(x.end_ms - x.start_ms for x in ss)
            parts.append(f"{cid} {_dev(cands, cid)} x{len(ss)}  slowest {_fmt_ms(slow)}  ${sum(x.cost for x in ss):.4f}")
        stage_end = max(s.end_ms for s in maps)
        out.append((stage_end - t0, "MAP    " + "  |  ".join(parts)))
        why = Counter(s.note.split(" [")[0] for s in maps)
        out.append((stage_end - t0, "       routes: " + ", ".join(f"{k} ×{v}" for k, v in why.most_common())))
    fails = [s for s in rec.spans if not s.ok]
    for s in fails:
        out.append((s.end_ms - t0, f"RETRY  {s.candidate_id}: {s.note}"))
    for s in rec.spans:
        if s.role in ("direct", "aggregator") and s.ok:
            out.append((s.end_ms - t0, f"{s.role.upper():6s} {s.candidate_id} {_dev(cands, s.candidate_id):18s} "
                                       f"{_fmt_ms(s.end_ms - s.start_ms):>10s}   ${s.cost:.4f}   {s.tokens_in:,} tokens in"))
        if s.role == "reduce":
            out.append((s.end_ms - t0, f"REDUCE {s.note}"))
    ans = list(rec.answer.values())[0] if isinstance(rec.answer, dict) and rec.answer else rec.answer
    out.append((rec.latency_ms, f"ANSWER {ans}   truth {rec.truth}   {'correct' if rec.correct else 'WRONG'}   "
                                f"total {_fmt_ms(rec.latency_ms)}   ${rec.cost:.4f}"))
    if base is not None:
        bans = list(base.answer.values())[0] if isinstance(base.answer, dict) and base.answer else base.answer
        trunc = "  (corpus truncated to fit window)" if base.truncated else ""
        out.append((rec.latency_ms, f"BASELINE ({base.pipeline}, {base.spans[-1].candidate_id if base.spans else '?'} direct)   "
                                    f"{_fmt_ms(base.latency_ms)}   ${base.cost:.4f}   answer {bans}   "
                                    f"{'correct' if base.correct else 'wrong'}{trunc}"))
    return out


def _dev(cands: dict, cid: str | None) -> str:
    if not cid or cid not in cands:
        return ""
    c = cands[cid]
    return {"groq": "groq-lpu", "cerebras": "cerebras-wse", "local": "cpu"}.get(c.provider, c.provider) + "-" + c.model.split("/")[-1][:10]


def demo(root: Path, cfg: dict, question: str, from_cache: bool = True, speed: str = "1x", run: str = "main-P3") -> int:
    from mosaic.config import load_candidates
    cands = load_candidates(root / "candidates.yaml")
    runs = root / cfg["runs_dir"]
    path = runs / f"{run}.jsonl.gz"
    if not path.exists():
        print(f"no run file {path}; run the evaluation first (make eval)", file=sys.stderr)
        return 1
    _, recs = read_run(path)
    rec = next((r for r in recs if r.question_id == question), None) if question else None
    if rec is None:
        rec = next((r for r in recs if r.cls == "A" and r.size == "128k" and r.correct), recs[0])
    base = None
    bp = runs / "main-P0.jsonl.gz"
    if bp.exists():
        base = next((r for r in read_run(bp)[1] if r.question_id == rec.question_id), None)
    if not from_cache:
        rec = asyncio.run(_live(root, cfg, rec.question_id)) or rec
    text = next((s.note for s in rec.spans if s.role == "question"), rec.question_id)
    sp = _speed(speed)
    shown = 0.0
    for t, line in sorted(render(rec, base, cands, text), key=lambda x: x[0]):
        if sp:
            time.sleep(max(0.0, min(3.0, (t - shown) / 1000 / sp)))
            shown = t
        print(line, flush=True)
    return 0


async def _live(root: Path, cfg: dict, qid: str):
    """Re-run the question now through P3 with the configured backends (simulated by default)."""
    from mosaic.data.store import load_corpus, load_questions
    from mosaic.pipelines import RunContext, run_question
    from mosaic.session import Session
    from mosaic.trace import Tracer
    s = Session(cfg, root=root)
    q = next(q for q in load_questions(s.data_dir, "eval") if q.question_id == qid)
    tr = Tracer("demo-live")
    gw = s.gateway(s.router("P3"), {"default": "sim"}, tr, live=False)
    ctx = RunContext(gw, "demo-live", {q.size: load_corpus(s.data_dir, "eval", q.size)}, planner=cfg["planner"])
    return await run_question("P3", q, ctx, 0.0)
