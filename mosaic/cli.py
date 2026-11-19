"""`mosaic` command line: the CLI and the report are the product (PRD 12).

Exit codes: 0 success, 1 user error, 2 provider failure after retries, 3 evaluation inconclusive.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from mosaic.config import ROOT, load_candidates, load_config

EXIT_OK, EXIT_USER, EXIT_PROVIDER, EXIT_INCONCLUSIVE = 0, 1, 2, 3


def _sizes(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _backends(live: Optional[str], replay: Optional[str] = None) -> dict[str, str]:
    b = {"default": "sim"}
    for cid in _sizes(live or ""):
        b[cid] = "live"
    for cid in _sizes(replay or ""):
        b[cid] = "replay"
    return b


def _select(questions, sizes, n_per_class, classes="LAP"):
    out = []
    for size in sizes:
        for cls in classes:
            sel = [q for q in questions if q.size == size and q.cls == cls]
            out += sel[:n_per_class] if n_per_class else sel
    return out


# ---- gen ---------------------------------------------------------------------------------
def cmd_gen(a, cfg) -> int:
    from mosaic.data.store import generate_all
    sizes = _sizes(a.sizes)
    print(f"generating corpora {sizes} seed={a.seed} into {a.data_dir} ...")
    m = generate_all(Path(a.data_dir), sizes, a.seed, a.n_per_class, a.n_profiling)
    for name, h in m["files"].items():
        print(f"  {name:36s} {h[:12]}")
    print(f"data hash {m['data_hash']}")
    return EXIT_OK


# ---- candidates ----------------------------------------------------------------------------
def cmd_candidates(a, cfg) -> int:
    import httpx
    from mosaic.config import resolve, save_candidates
    cands = load_candidates()
    providers = _sizes(a.providers)
    for prov in providers:
        pc = cfg["providers"].get(prov, {})
        base = pc.get("base_url")
        try:
            key = resolve(pc.get("key", "")) if pc.get("key") else ""
            r = httpx.get(f"{base.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"} if key else {},
                          timeout=10)
            r.raise_for_status()
            models = {m["id"]: m for m in r.json().get("data", [])}
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{prov}: not reachable ({type(e).__name__}); keeping candidates.yaml rows as they are")
            continue
        print(f"{prov}: {len(models)} models listed")
        for c in cands.values():
            if c.provider != prov:
                continue
            m = models.get(c.model)
            if m is None:
                print(f"  WARNING {c.id} {c.model} is no longer listed")
                continue
            cw = m.get("context_window") or m.get("max_context_length")
            if cw and int(cw) != c.context_window:
                print(f"  {c.id}: context window {c.context_window} -> {cw}")
                c.context_window = int(cw)
            if a.check_limits and prov != "local":
                _check_limits(base, key, c)
    if not a.dry_run:
        save_candidates(cands)
        print("wrote candidates.yaml")
    return EXIT_OK


def _check_limits(base: str, key: str, c) -> None:
    import httpx
    body = {"model": c.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=body,
                   headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=20)
    lim_t, lim_r = r.headers.get("x-ratelimit-limit-tokens"), r.headers.get("x-ratelimit-limit-requests")
    if lim_t and c.tpm and int(lim_t) != c.tpm:
        print(f"  WARNING {c.id}: provider says {lim_t} tokens/min, candidates.yaml says {c.tpm}")
    if lim_r and int(lim_r) not in (c.rpm, c.rpd):
        print(f"  WARNING {c.id}: provider request limit {lim_r} matches neither rpm={c.rpm} nor rpd={c.rpd}")


# ---- run -----------------------------------------------------------------------------------
def estimate_cost(questions, pipeline, corpora, cands, baseline) -> tuple[int, float]:
    from mosaic.core.executor import apply_filter, chunk, parse_log
    from mosaic.core.planner import reference_plan
    from mosaic.tokens import count_tokens
    c = cands[baseline]
    tokens = 0
    for q in questions:
        lines = [t.line() for t in corpora[q.size]]
        if pipeline == "P0":
            tokens += min(sum(count_tokens(x) for x in lines), c.context_window)
            continue
        plan = reference_plan(q)
        recs, _ = apply_filter(parse_log(lines), plan)
        tokens += 2500 + sum(count_tokens(r.line) for r in recs) + 700 * max(1, len(chunk(recs, plan.chunk_tokens)))
    return tokens, tokens * (c.price_in + c.price_out) / 2 / 1e6


def cmd_run(a, cfg) -> int:
    from mosaic.data.store import data_hash, load_corpus, load_questions
    from mosaic.gateway import SpendCapExceeded
    from mosaic.pipelines import RunContext, run_meta, run_pipeline, write_run
    from mosaic.session import Session
    from mosaic.trace import Tracer

    s = Session(cfg)
    if not (s.data_dir / f"questions-{a.split}.jsonl").exists():
        print("no data: run `mosaic gen` first", file=sys.stderr)
        return EXIT_USER
    sizes = _sizes(a.sizes)
    qs = _select(load_questions(s.data_dir, a.split), sizes, a.n_per_class, a.classes)
    if a.question:
        qs = [q for q in load_questions(s.data_dir, a.split) if q.question_id in _sizes(a.question)]
    corpora = {sz: load_corpus(s.data_dir, a.split, sz) for sz in sorted({q.size for q in qs})}
    backends = _backends(a.live, a.replay)
    pool = _sizes(a.pool) if a.pool else cfg["main_pool"]
    if a.fixed:
        pool = [a.fixed]
    if a.budget_ms:
        cfg["latency_budget_ms"] = a.budget_ms
    live = any(v == "live" for k, v in backends.items() if k != "default")
    run_id = a.run_id or f"{a.pipeline}-{'live' if live else 'sim'}-{'-'.join(sizes)}" + (f"-{a.fixed}" if a.fixed else "")
    out = s.runs_dir / f"{run_id}.jsonl.gz"
    if a.resume and out.exists() and not a.no_cache:
        print(f"{out} exists; --resume replays cached calls and only sends what is missing")
    tokens, usd = estimate_cost(qs, a.pipeline, corpora, s.cands, pool[0] if a.fixed else cfg["baseline"])
    print(f"run {run_id}: {len(qs)} questions, pipeline {a.pipeline}, pool {pool}, backends {backends}")
    print(f"estimated ~{tokens:,} tokens, ~${usd:.4f} at list price" + (" (simulated: no spend)" if not live else ""))
    if a.dry_run:
        return EXIT_OK
    if live and usd > cfg["confirm_above_usd"] and not a.yes:
        if input(f"estimated spend ${usd:.2f} exceeds ${cfg['confirm_above_usd']}; continue? [y/N] ").lower() != "y":
            return EXIT_USER
    tracer = Tracer(run_id)
    router = s.router(a.pipeline, pool=pool, explore=a.explore)
    gw = s.gateway(router, backends, tracer, live=live, use_cache=True, read_cache=not a.no_cache,
                   spend_cap=cfg["spend_cap_usd"] if live else None)
    t0 = time.time()
    if a.plans_only:
        return _plans_only(qs, corpora, gw, ctx_planner=a.planner or cfg["planner"], sample=cfg.get("planner_sample_tokens", 800))

    def progress(i, n, rec):
        if i % max(1, a.progress_every) == 0 or i == n:
            heads = " ".join(f"{cid}:{gw.limiters[cid].headroom(gw.clock.now()):.0%}" for cid in pool if cid in gw.limiters)
            print(f"  [{i}/{n}] {rec.question_id} {'ok ' if rec.correct else 'BAD'} {rec.latency_ms:8.0f} ms "
                  f"spend ${gw.spent:.4f} headroom {heads} cache {gw.cache.hits if gw.cache else 0}h", flush=True)

    ctx = RunContext(gw, run_id, corpora, planner=a.planner or cfg["planner"], arrival_gap_ms=cfg["arrival_gap_ms"],
                     max_depth=cfg["max_depth"], progress=progress)
    try:
        recs = asyncio.run(run_pipeline(a.pipeline, qs, ctx))
    except SpendCapExceeded as e:
        print(f"stopped: {e}", file=sys.stderr)
        return EXIT_PROVIDER
    meta = run_meta(run_id, a.pipeline, cfg, s.cands, data_hash(s.data_dir), backends, pool=pool,
                    router=router.mode, planner=ctx.planner, split=a.split, clock="real" if live else "virtual",
                    explore=router.explore_rate)
    write_run(out, meta, recs)
    _summary(recs, gw, time.time() - t0)
    print(f"wrote {out}")
    failed = sum(1 for r in recs for sp in r.spans if not sp.ok)
    return EXIT_PROVIDER if recs and all(r.answer is None for r in recs) and failed else EXIT_OK


def _plans_only(qs, corpora, gw, ctx_planner, sample) -> int:
    """Populate the response cache with plans so later runs share them exactly."""
    from mosaic.core.planner import make_plan
    from mosaic.pipelines import sample_lines

    async def go():
        for q in qs:
            lines = [t.line() for t in corpora[q.size]]
            plan, _, notes = await make_plan(q.text, sample_lines(lines, sample), gw, gw.clock.now(), q.question_id,
                                             pin=ctx_planner)
            print(f"  {q.question_id}: {plan.strategy} reduce={plan.reduce} semantic={plan.filter.semantic} {notes or ''}",
                  flush=True)
    asyncio.run(go())
    return EXIT_OK


def _summary(recs, gw, secs) -> None:
    by = Counter()
    for r in recs:
        for sp in r.spans:
            if sp.candidate_id:
                by[(sp.candidate_id, sp.ok)] += 1
    acc = sum(r.correct for r in recs) / max(1, len(recs))
    print(f"accuracy {acc:.1%} over {len(recs)}; cost ${sum(r.cost for r in recs):.4f}; {secs:.1f}s wall")
    for (cid, ok), n in sorted(by.items()):
        print(f"  {cid} {'ok' if ok else 'failed'}: {n} calls")
    print(f"  fallbacks {gw.fallbacks}; cache hits {gw.cache.hits if gw.cache else 0}")


# ---- profile / train / calibrate ----------------------------------------------------------
def cmd_profile(a, cfg) -> int:
    from mosaic.routing.profiler import run_profile
    return run_profile(a, cfg)


def cmd_train(a, cfg) -> int:
    from mosaic.routing.profiler import load_labels
    from mosaic.routing.predictor import train
    rows = load_labels(ROOT / "profiles" / "labels.jsonl")
    if not rows:
        print("no profiling labels: run `mosaic profile` first", file=sys.stderr)
        return EXIT_USER
    if a.role != "all":
        rows = [r for r in rows if r["role"] == a.role]
    pred = train(rows, seed=a.seed)
    out = ROOT / "router" / "model.json"
    pred.save(out)
    print(f"{'role|candidate':16s} {'n':>5s} {'base':>6s} {'brier':>7s} {'const':>7s} {'beats':>6s} {'iso':>4s} {'gap':>5s}")
    for k, r in sorted(pred.report.items()):
        print(f"{k:16s} {r['n_train'] + r['n_test']:5d} {r['base_rate']:6.3f} {r['brier']:7.4f} "
              f"{r['brier_constant']:7.4f} {str(r['beats_constant']):>6s} {'y' if r['calibrated_isotonic'] else 'n':>4s} "
              f"{r['max_bin_gap']:5.2f}")
    print(f"wrote {out}")
    return EXIT_OK


def cmd_calibrate(a, cfg) -> int:
    from mosaic.routing.profiler import calibrate_sim
    return calibrate_sim(a, cfg)


# ---- eval / report / demo -------------------------------------------------------------------
def cmd_eval(a, cfg) -> int:
    from mosaic.eval.compare import WORDING, compare
    from mosaic.pipelines import read_run
    runs = ROOT / cfg["runs_dir"]
    _, ra = read_run(runs / f"{a.a}.jsonl.gz")
    _, rb = read_run(runs / f"{a.b}.jsonl.gz")
    res = compare(ra, rb, margin=a.margin, pooled={"primary": (("A", "P"), ("128k", "512k"))})
    print(f"B={a.b} vs A={a.a}, margin {a.margin} points, one-sided {1 - a.alpha:.0%} paired bootstrap")
    print(f"{'cell':12s} {'n':>4s} {'acc A':>6s} {'acc B':>6s} {'diff':>6s} {'CI low':>7s} {'CI high':>7s} {'p_holm':>7s}  verdict")
    rows = res["cells"] + res.get("primary", [])
    for v in rows:
        print(f"{v.cls + ' ' + v.size:12s} {v.n:4d} {v.acc_a:6.1%} {v.acc_b:6.1%} {v.diff:+6.1f} {v.ci_low:+7.1f} "
              f"{v.ci_high:+7.1f} {v.p_adj if v.p_adj is not None else v.p_value:7.3f}  {v.outcome}: "
              f"{WORDING[v.outcome].format(m=a.margin)} ({v.why})")
    prim = res.get("primary", [])
    return EXIT_INCONCLUSIVE if prim and prim[0].outcome == "inconclusive" else EXIT_OK


def cmd_report(a, cfg) -> int:
    from mosaic.report.report import build_report
    path = build_report(ROOT, cfg)
    print(f"wrote {path}")
    if a.open:
        print((path).read_text())
    return EXIT_OK


def cmd_demo(a, cfg) -> int:
    from mosaic.report.demo import demo
    return demo(ROOT, cfg, a.question, from_cache=a.from_cache, speed=a.speed, run=a.run)


def main(argv: Optional[list[str]] = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(prog="mosaic", description="Heterogeneous recursion router")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate corpora and question sets")
    g.add_argument("--sizes", default=",".join(cfg["eval"]["sizes"]))
    g.add_argument("--seed", type=int, default=cfg["seed"])
    g.add_argument("--n-per-class", type=int, default=cfg["eval"]["n_per_class"])
    g.add_argument("--n-profiling", type=int, default=cfg["eval"]["profiling_questions"])
    g.add_argument("--data-dir", default=str(ROOT / cfg["data_dir"]))
    g.set_defaults(fn=cmd_gen)

    c = sub.add_parser("candidates", help="discover models and refresh candidates.yaml")
    c.add_argument("--providers", default="groq,cerebras,local")
    c.add_argument("--check-limits", action="store_true", help="send a 1-token call to read rate-limit headers")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_candidates)

    pr = sub.add_parser("profile", help="measure latency, speed, errors and quality per candidate")
    pr.add_argument("--set", default="profiling")
    pr.add_argument("--concurrency", default="1,max")
    pr.add_argument("--candidates", default="C4,C5,C6")
    pr.add_argument("--live", default="", help="candidate ids to call live (others simulated)")
    pr.add_argument("--n-questions", type=int, default=0)
    pr.add_argument("--subset", default="C4=100", help="per-candidate question caps, e.g. C4=100 (C4's daily quota)")
    pr.add_argument("--from-runs", default="", help="also import oracle-labelled spans from these live runs")
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(fn=cmd_profile)

    t = sub.add_parser("train-router", help="fit the quality predictor")
    t.add_argument("--role", default="all")
    t.add_argument("--seed", type=int, default=cfg["seed"])
    t.set_defaults(fn=cmd_train)

    cal = sub.add_parser("calibrate", help="fit simulator error rates from live, oracle-labelled runs")
    cal.add_argument("--runs", required=True, help="comma-separated live run ids")
    cal.add_argument("--dry-run", action="store_true")
    cal.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("run", help="run a pipeline on a question set")
    r.add_argument("--pipeline", required=True, choices=["P0", "P1", "P2", "P3"])
    r.add_argument("--sizes", default=",".join(cfg["eval"]["sizes"]))
    r.add_argument("--split", default="eval", choices=["eval", "profiling"])
    r.add_argument("--classes", default="LAP")
    r.add_argument("--n-per-class", type=int, default=0)
    r.add_argument("--question", default="")
    r.add_argument("--budget-ms", type=float, default=None)
    r.add_argument("--explore", type=float, default=None)
    r.add_argument("--pool", default="")
    r.add_argument("--fixed", default="", help="pin every executor call to one candidate (device study)")
    r.add_argument("--planner", default="")
    r.add_argument("--live", default="", help="candidate ids served by real APIs; the rest are simulated")
    r.add_argument("--replay", default="", help="candidate ids served only from the response cache (offline)")
    r.add_argument("--run-id", default="")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--yes", action="store_true")
    r.add_argument("--plans-only", action="store_true", help="only run the planner (fills the cache)")
    r.add_argument("--progress-every", type=int, default=10)
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="compare two runs")
    e.add_argument("--a", required=True)
    e.add_argument("--b", required=True)
    e.add_argument("--margin", type=float, default=cfg["margin_points"])
    e.add_argument("--alpha", type=float, default=cfg["alpha"])
    e.set_defaults(fn=cmd_eval)

    rp = sub.add_parser("report", help="build report/report.md from run records")
    rp.add_argument("--from-cache", action="store_true", help="(always true: the report never calls a provider)")
    rp.add_argument("--open", action="store_true")
    rp.set_defaults(fn=cmd_report)

    d = sub.add_parser("demo", help="show one question end to end")
    d.add_argument("--question", default="")
    d.add_argument("--from-cache", action="store_true")
    d.add_argument("--speed", default="1x")
    d.add_argument("--run", default="main-P3")
    d.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    try:
        return a.fn(a, cfg)
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return EXIT_USER
    except KeyboardInterrupt:
        return EXIT_USER


if __name__ == "__main__":
    sys.exit(main())
