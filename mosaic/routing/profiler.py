"""Profiler: per-candidate latency, speed, errors and oracle-labelled quality (PRD 9.6), plus
`calibrate`, which fits the simulator's error model from live, labelled map calls."""
from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mosaic.config import ROOT, load_candidates, save_candidates
from mosaic.core.planner import make_plan, reference_plan
from mosaic.providers.base import ProviderError
from mosaic.types import Profile, RunRecord, Span

LABELS = ROOT / "profiles" / "labels.jsonl"
PROFILES = ROOT / "profiles" / "profiles.parquet"


def summarize(spans: Iterable[Span], candidate_id: str, concurrency: int, source: str) -> Profile | None:
    """Latency profile from call spans. Service time is fitted as
    total = base + tokens_in / prefill_tps + tokens_out / tps_out (non-negative least squares),
    which also works for servers that buffer the whole answer into one chunk (TTFT = total)."""
    from scipy.optimize import nnls
    sp = [s for s in spans if s.candidate_id == candidate_id and s.role != "question"]
    ok = [s for s in sp if s.ok]
    if not ok:
        return None
    ttft = np.array([s.ttft_ms for s in ok])
    X = np.array([[1.0, s.tokens_in, s.tokens_out] for s in ok])
    y = np.array([s.total_ms for s in ok])
    if len(ok) >= 5 and np.linalg.matrix_rank(X) == 3:
        coef, _ = nnls(X, y)
        base, b_in, b_out = coef
    else:
        base, b_in, b_out = float(np.median(ttft)), 0.0, 0.0
    return Profile(candidate_id=candidate_id, ttft_p50=float(np.percentile(ttft, 50)), ttft_p95=float(np.percentile(ttft, 95)),
                   ttft_base=float(base), prefill_tps=float(1000 / b_in) if b_in > 1e-6 else 1e6,
                   tps_out=float(1000 / b_out) if b_out > 1e-6 else 1e6, error_rate=1 - len(ok) / len(sp),
                   n_calls=len(sp), concurrency=concurrency,
                   measured_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"), source=source)


def labels_from_records(recs: list[RunRecord], source: str) -> list[dict]:
    """Router training rows: map spans carry their own oracle label; a direct span is labelled
    by the question's correctness (it is the only model call)."""
    rows = []
    for r in recs:
        for s in r.spans:
            if s.candidate_id is None or not s.ok:
                continue
            if s.role == "mapper" and s.correct is not None:
                rows.append({"role": "mapper", "candidate_id": s.candidate_id, "features": s.features,
                             "correct": bool(s.correct), "question_id": r.question_id, "source": source})
            elif s.role == "direct":
                rows.append({"role": "direct", "candidate_id": s.candidate_id, "features": s.features,
                             "correct": bool(r.correct), "question_id": r.question_id, "source": source})
    return rows


def plan_equivalent(a: dict, b: dict) -> bool:
    keys = ("strategy", "reduce", "reduce_args")
    if any(a.get(k) != b.get(k) for k in keys):
        return False
    fa, fb = a.get("filter", {}), b.get("filter", {})
    return fa.get("fields") == fb.get("fields") and (a["strategy"] == "direct" or fa.get("semantic") == fb.get("semantic")
                                                     or not fb.get("keywords"))


def load_labels(path: Path = LABELS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_profile(a, cfg, root: Path = ROOT) -> int:
    from mosaic.data.store import load_corpus, load_questions
    from mosaic.pipelines import RunContext, read_run, run_pipeline, sample_lines
    from mosaic.session import Session
    from mosaic.trace import Tracer

    s = Session(cfg, root=root)
    labels_path, profiles_path = root / "profiles" / "labels.jsonl", root / "profiles" / "profiles.parquet"
    qs = load_questions(s.data_dir, a.set)
    if a.n_questions:
        qs = qs[: a.n_questions]
    corpora = {sz: load_corpus(s.data_dir, a.set, sz) for sz in sorted({q.size for q in qs})}
    cands = [c for c in a.candidates.split(",") if c]
    live = set(x for x in a.live.split(",") if x)
    levels = a.concurrency.split(",")
    print(f"profiling {cands} on {len(qs)} {a.set} questions at concurrency {levels}; live: {sorted(live) or 'none'}")
    if a.dry_run:
        return 0
    profiles, labels = [], []
    measured: set[str] = set()
    for rid in [x for x in a.from_runs.split(",") if x]:
        meta, recs = read_run(s.runs_dir / f"{rid}.jsonl.gz")
        spans = [sp for r in recs for sp in r.spans]
        for cid in meta.get("pool", []):  # the run's own candidate; planner replays are not profiled again
            prof = summarize(spans, cid, 1, f"live:{rid}")
            if prof and meta.get("backends", {}).get(cid) == "live":
                profiles.append(prof)
                c = s.cands[cid]  # measured latency replaces the prior, so simulated calls match it
                c.ttft_ms, c.prefill_tps, c.tps_out = (round(prof.ttft_base, 1), round(prof.prefill_tps, 1),
                                                       round(prof.tps_out, 1))
                measured.add(cid)
        labels += [x for x in labels_from_records(recs, f"live:{rid}")
                   if meta.get("backends", {}).get(x["candidate_id"]) == "live" and x["candidate_id"] in meta.get("pool", [])]
    if measured and not a.dry_run:
        save_candidates(s.cands, root / "candidates.yaml")
        print(f"latency priors for {sorted(measured)} replaced by live measurements in candidates.yaml")
    caps = {k: int(v) for k, v in (x.split("=") for x in a.subset.split(",") if x)}
    all_qs = qs
    for cid in cands:
        qs = all_qs[: caps[cid]] if cid in caps else all_qs
        if cid in caps:  # keep the class/size mix: take every k-th question
            step = max(1, len(all_qs) // caps[cid])
            qs = all_qs[::step][: caps[cid]]
        for lvl in levels:
            if lvl != "1" and s.cands[cid].max_concurrency == 1:
                continue  # "max" is the same as 1 for a single-slot server
            backends = {"default": "sim", cid: "live"} if cid in live else {"default": "sim"}
            tracer = Tracer(f"profile-{cid}-{lvl}")
            router = s.router("P1", pool=[cid])
            gw = s.gateway(router, backends, tracer, live=cid in live, use_cache=True)
            gw.profiles = {}
            if lvl == "1":
                gw.limiters[cid].max_concurrency = 1
            ctx = RunContext(gw, tracer.run_id, corpora, use_reference_plans=True, arrival_gap_ms=cfg["arrival_gap_ms"])
            recs = asyncio.run(run_pipeline("P1", qs, ctx))
            spans = [sp for r in recs for sp in r.spans]
            prof = summarize(spans, cid, 1 if lvl == "1" else gw.limiters[cid].max_concurrency, "live" if cid in live else "sim")
            if prof:
                profiles.append(prof)
            if lvl == levels[-1] or (s.cands[cid].max_concurrency == 1 and lvl == "1"):
                labels += labels_from_records(recs, "live" if cid in live else "sim")
                failed = sum(1 for r in recs for n in r.notes if "failed on every candidate" in n)
                if failed:
                    print(f"  {cid}: {failed} map calls could not be served (window, quota or errors)")
        # planner labels: does this candidate's plan match the reference plan?
        if "planner" in s.cands[cid].roles:
            tracer = Tracer(f"profile-plan-{cid}")
            gw = s.gateway(s.router("P1", pool=[cid]), {"default": "sim", cid: "live"} if cid in live else {"default": "sim"},
                           tracer, live=cid in live)

            async def plans():
                out = []
                for q in qs:
                    lines = [t.line() for t in corpora[q.size]]
                    try:
                        p, _, _ = await make_plan(q.text, sample_lines(lines, 800), gw, gw.clock.now(), q.question_id, pin=cid)
                        ok = plan_equivalent(p.model_dump(), reference_plan(q).model_dump())
                    except (ValueError, ProviderError):
                        ok = False
                    out.append({"role": "planner", "candidate_id": cid, "features": {}, "correct": ok,
                                "question_id": q.question_id, "source": "live" if cid in live else "sim"})
                return out
            labels += asyncio.run(plans())
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([p.model_dump() for p in profiles])
    if profiles_path.exists():
        old = pd.read_parquet(profiles_path)
        keep = ~old.set_index(["candidate_id", "concurrency", "source"]).index.isin(
            df.set_index(["candidate_id", "concurrency", "source"]).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    df = df.sort_values(["candidate_id", "concurrency", "source"]).reset_index(drop=True)
    df.to_parquet(profiles_path, index=False)
    old_labels = [x for x in load_labels(labels_path) if (x["candidate_id"], x["source"]) not in {(y["candidate_id"], y["source"]) for y in labels}]
    labels_path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in old_labels + labels))
    print(df[["candidate_id", "concurrency", "source", "ttft_p50", "ttft_p95", "prefill_tps", "tps_out", "error_rate",
              "n_calls"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    by = defaultdict(list)
    for x in labels:
        by[(x["role"], x["candidate_id"])].append(x["correct"])
    for (role, cid), ys in sorted(by.items()):
        print(f"  labels {role:8s} {cid}: n={len(ys)} correct={np.mean(ys):.3f}")
    return 0


# ---- simulator calibration -----------------------------------------------------------------
def ticket_errors(world, events: list[dict], cands=None) -> dict[str, dict[str, list[int]]]:
    """events: {candidate_id, lines, rows, schema, malformed}. Returns, per weights family, the
    (errors, observations) behind each error-model parameter. Identical calls (cache replays
    shared by several questions) are counted once."""
    from mosaic.providers.sim import KIND_PARAM, field_kind, truth_of
    from mosaic.tokens import count_tokens
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cands = cands or load_candidates()
    seen = set()
    for ev in events:
        key = (cands[ev["candidate_id"]].sim.get("weights", ev["candidate_id"]), tuple(ev["lines"]),
               json.dumps(ev["schema"], sort_keys=True), ev.get("task", ""))
        if key in seen:
            continue
        seen.add(key)
        d = out[key[0]]
        d["malformed"][0] += int(ev["malformed"])
        d["malformed"][1] += 1
        if ev["malformed"]:
            continue
        d["rot"][0] += int(1000 * (1 + sum(count_tokens(x) for x in ev["lines"]) / 6000))
        d["rot"][1] += 1000
        by = {r["ticket_id"]: r for r in ev["rows"]}
        for ln in ev["lines"]:
            t = world.by_line.get(ln)
            if t is None:
                continue
            r = by.get(t.ticket_id)
            d["drop"][0] += int(r is None)
            d["drop"][1] += 1
            if r is None:
                continue
            for name, typ in ev["schema"].items():
                kind = field_kind(name, typ)
                if kind not in KIND_PARAM or (kind == "category" and restricted(ev.get("task", ""))):
                    continue
                truth, v = truth_of(t, kind), r.get(name)
                wrong = (abs(float(v) - truth) > 0.01 if (truth is not None and v is not None)
                         else (truth is None) != (v is None)) if kind == "amount" else v != truth
                d[KIND_PARAM[kind]][0] += int(wrong)
                d[KIND_PARAM[kind]][1] += 1
    return out


def restricted(task: str) -> bool:
    """A map task that names some categories but not all five asks a narrower question; its
    'other' answers are not category errors, so those observations are skipped."""
    named = [c for c in ("billing", "shipping", "defect", "account", "other") if c in task.lower()]
    return 0 < len(named) < 5


PRIOR_STRENGTH = 50  # pseudo-observations: a parameter with little live data stays near its prior


def fit_params(counts: dict[str, list[int]], priors: dict[str, float]) -> dict[str, float]:
    """Posterior-mean rates under a Beta prior centred on the current value, normalised by the
    simulator's context-rot factor (1 + chunk tokens / 6000) for per-ticket rates."""
    rot = counts["rot"][0] / counts["rot"][1] if counts.get("rot", [0, 0])[1] else 1.5
    out: dict[str, float] = {}
    for k in ("err_category", "err_sentiment", "err_refund", "drop", "malformed"):
        e, n = counts.get(k, [0, 0])
        if not n:
            continue
        scale = 1.0 if k == "malformed" else rot
        p0 = priors[k] * scale
        out[k] = round((e + PRIOR_STRENGTH * p0) / (n + PRIOR_STRENGTH) / scale, 5)
        out[f"{k}_n"] = n
        out[f"{k}_e"] = e
    return out


def calibrate_sim(a, cfg, root: Path = ROOT) -> int:
    from mosaic.core.executor import Executor
    from mosaic.pipelines import read_run
    from mosaic.session import Session

    s = Session(cfg, root=root)
    events: list[dict] = []
    orig = Executor.map_chunk

    async def capture(self, recs, plan, at, qid, parent, depth, res):
        rows, end = await orig(self, recs, plan, at, qid, parent, depth, res)
        spans = [sp for sp in self.gw.tracer.spans.get(qid, []) if sp.role == "mapper" and sp.ok]
        if spans:
            events.append({"candidate_id": spans[-1].candidate_id, "lines": [r.line for r in recs], "rows": rows,
                           "schema": plan.map_schema, "task": plan.map_task, "malformed": not rows and bool(recs)})
        return rows, end

    from mosaic.pipelines import RunContext, run_pipeline
    from mosaic.trace import Tracer
    from mosaic.data.store import load_corpus, load_questions
    Executor.map_chunk = capture
    try:
        for rid in a.runs.split(","):
            meta, recs = read_run(s.runs_dir / f"{rid}.jsonl.gz")
            qids = {r.question_id for r in recs}
            qs = [q for q in load_questions(s.data_dir, meta.get("split", "eval")) if q.question_id in qids]
            corpora = {sz: load_corpus(s.data_dir, meta.get("split", "eval"), sz) for sz in {q.size for q in qs}}
            backends = {k: ("replay" if v == "live" else v) for k, v in meta["backends"].items()}
            tracer = Tracer(rid)
            gw = s.gateway(s.router("P1", pool=meta["pool"]), backends, tracer, live=False)
            ctx = RunContext(gw, rid, corpora, planner=meta.get("planner"))
            asyncio.run(run_pipeline(meta["pipeline"], qs, ctx))
            print(f"replayed {rid}: {len(qs)} questions, cache hits {gw.cache.hits}, misses {gw.cache.misses}")
    finally:
        Executor.map_chunk = orig
    counts = ticket_errors(s.world, events, s.cands)
    cands = s.cands
    print(f"{'weights':20s} {'param':13s} {'prior':>8s} {'fitted':>8s} {'errors':>7s} {'n':>6s}")
    for w, cnt in sorted(counts.items()):
        fam = [c for c in cands.values() if c.sim.get("weights") == w]
        base = fam[0].sim
        mean_err = (base.get("prior_err_easy", base.get("err_easy", 0.003)) + base.get("prior_err_hard", base.get("err_hard", 0.02))) / 2
        priors = {"err_category": mean_err, "err_sentiment": mean_err, "err_refund": mean_err,
                  "drop": base.get("drop", 0.002), "malformed": base.get("malformed", 0.005)}
        priors = {k: base.get(f"prior_{k}", v) for k, v in priors.items()}
        fitted = fit_params(cnt, priors)
        for k in ("err_category", "err_sentiment", "err_refund", "drop", "malformed"):
            if k in fitted:
                print(f"{w:20s} {k:13s} {priors[k]:8.4f} {fitted[k]:8.4f} {fitted[k + '_e']:7d} {fitted[k + '_n']:6d}")
        for c in fam:
            for k in priors:
                c.sim.setdefault(f"prior_{k}", round(priors[k], 5))
                if k in fitted:
                    c.sim[k] = fitted[k]
                    c.sim[f"{k}_n"] = fitted[f"{k}_n"]
            c.sim["calibrated_from"] = a.runs
    if not a.dry_run:
        save_candidates(cands, root / "candidates.yaml")
        print("updated sim parameters in candidates.yaml")
    return 0
