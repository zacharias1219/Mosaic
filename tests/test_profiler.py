"""9.6 Profiler, simulator calibration, config and data store."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from mosaic.config import ROOT, build_clients, load_candidates, load_config, resolve, save_candidates
from mosaic.core.planner import reference_plan
from mosaic.data.store import data_hash, generate_all, load_corpus, load_questions
from mosaic.routing.profiler import (calibrate_sim, fit_params, labels_from_records, load_labels, plan_equivalent,
                                     restricted, run_profile, summarize, ticket_errors)
from mosaic.types import RunRecord, Span


@pytest.fixture(scope="module")
def mini_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mosaic")
    for f in ("mosaic.yaml", "candidates.yaml"):
        shutil.copy(ROOT / f, root / f)
    generate_all(root / "data", ["8k"], 42, 4, 12)
    return root


def span(cid, tin, tout, total, ttft=None, ok=True, **kw):
    return Span(run_id="r", question_id="q", span_id=f"q:{tin}:{tout}", role="mapper", candidate_id=cid, tokens_in=tin,
                tokens_out=tout, total_ms=total, ttft_ms=ttft if ttft is not None else total, ok=ok, **kw)


def test_summarize_recovers_speeds_even_when_answer_is_buffered():
    # total = 200 + tin/5000 s + tout/400 s, delivered in one chunk (TTFT = total)
    spans = [span("X", tin, tout, 200 + 1000 * tin / 5000 + 1000 * tout / 400) for tin, tout in
             [(1000, 50), (3000, 400), (2000, 900), (500, 20), (4000, 1200), (2500, 300)]]
    spans.append(span("X", 100, 10, 0, ok=False))
    p = summarize(spans, "X", 1, "live")
    assert abs(p.ttft_base - 200) < 1 and abs(p.prefill_tps - 5000) < 5 and abs(p.tps_out - 400) < 1
    assert abs(p.error_rate - 1 / 7) < 1e-9 and p.n_calls == 7
    assert summarize(spans, "nobody", 1, "live") is None


def test_labels_from_records_uses_map_labels_and_direct_question_correctness():
    sp = [span("C5", 10, 5, 100, correct=True, features={"n_tickets": 3.0}),
          span("C5", 10, 5, 100, correct=None), span("C4", 10, 5, 100, ok=False, correct=False),
          Span(run_id="r", question_id="q", span_id="d", role="direct", candidate_id="C4", ok=True)]
    rec = RunRecord(run_id="r", pipeline="P1", question_id="q", cls="L", size="8k", correct=False, spans=sp)
    rows = labels_from_records([rec], "sim")
    assert [(r["role"], r["candidate_id"], r["correct"]) for r in rows] == [("mapper", "C5", True), ("direct", "C4", False)]


def test_plan_equivalence_and_restricted_tasks(tickets_8k):
    from mosaic.data.questions import build_questions
    q = build_questions(tickets_8k, "A", 5, 1, "8k")[0]
    ref = reference_plan(q).model_dump()
    assert plan_equivalent(ref, ref)
    other = {**ref, "reduce": "sum" if ref["reduce"] != "sum" else "count"}
    assert not plan_equivalent(other, ref)
    assert restricted("Classify each ticket as shipping, defect, or other.")
    assert not restricted("Classify the category of each ticket.")
    assert not restricted("Classify as billing, shipping, defect, account or other.")


def test_fit_params_shrinks_thin_data_toward_prior():
    counts = {"rot": [1500, 1000], "err_category": [0, 3], "err_refund": [30, 300], "malformed": [0, 10]}
    f = fit_params(counts, {"err_category": 0.01, "err_refund": 0.01, "err_sentiment": 0.01, "drop": 0.001,
                            "malformed": 0.005})
    assert abs(f["err_category"] - 0.01) < 0.002          # 3 observations barely move it
    assert 0.04 < f["err_refund"] < 0.07                  # 300 observations dominate (10% raw / rot 1.5)
    assert "err_sentiment" not in f and f["malformed"] < 0.005


def test_ticket_errors_dedupes_identical_calls(world_8k, tickets_8k):
    lines = [t.line() for t in tickets_8k[:5]]
    rows = [{"ticket_id": t.ticket_id, "category": t.category} for t in tickets_8k[:5]]
    rows[0]["category"] = "other" if tickets_8k[0].category != "other" else "billing"
    ev = {"candidate_id": "C5", "lines": lines, "rows": rows, "schema": {"category": "category"},
          "task": "Classify the category of each ticket.", "malformed": False}
    out = ticket_errors(world_8k, [ev, dict(ev)], load_candidates())
    assert out["qwen3.8-27b"]["err_category"] == [1, 5]


def test_run_profile_writes_profiles_and_labels(mini_root):
    a = argparse.Namespace(set="profiling", n_questions=0, candidates="C5,C6", live="", concurrency="1,max",
                           subset="C4=100", from_runs="", dry_run=False)
    assert run_profile(a, load_config(mini_root / "mosaic.yaml"), root=mini_root) == 0
    df = pd.read_parquet(mini_root / "profiles" / "profiles.parquet")
    assert set(df["candidate_id"]) == {"C5", "C6"} and set(df[df.candidate_id == "C5"]["concurrency"]) == {1, 16}
    labels = load_labels(mini_root / "profiles" / "labels.jsonl")
    roles = {(x["role"], x["candidate_id"]) for x in labels}
    assert ("mapper", "C5") in roles and ("planner", "C5") in roles and ("mapper", "C6") in roles


def test_calibrate_replays_offline_and_updates_candidates(mini_root, tmp_path):
    from mosaic.pipelines import RunContext, run_pipeline, run_meta, write_run
    from mosaic.session import Session
    from mosaic.trace import Tracer
    import asyncio
    cfg = load_config(mini_root / "mosaic.yaml")
    s = Session(cfg, root=mini_root)
    qs = [q for q in load_questions(s.data_dir, "eval") if q.cls in "AP"][:4]
    gw = s.gateway(s.router("P1", pool=["C5"]), {"default": "sim"}, Tracer("cal"), live=False)
    recs = asyncio.run(run_pipeline("P1", qs, RunContext(gw, "cal", {"8k": load_corpus(s.data_dir, "eval", "8k")},
                                                        planner="C5")))
    write_run(s.runs_dir / "cal.jsonl.gz", run_meta("cal", "P1", cfg, s.cands, "d", {"default": "sim"}, pool=["C5"],
                                                     planner="C5", split="eval"), recs)
    before = load_candidates(mini_root / "candidates.yaml")["C5"].sim
    assert calibrate_sim(argparse.Namespace(runs="cal", dry_run=False), cfg, root=mini_root) == 0
    after = load_candidates(mini_root / "candidates.yaml")["C5"].sim
    assert after["calibrated_from"] == "cal" and "err_category_n" in after and "prior_err_category" in after
    assert load_candidates(mini_root / "candidates.yaml")["C2"].sim["calibrated_from"] == "cal"  # same weights
    assert before != after


def test_store_and_config(mini_root, monkeypatch):
    assert data_hash(mini_root / "data") and len(load_questions(mini_root / "data", "eval")) == 12
    monkeypatch.setenv("X_URL", "https://x")
    assert resolve("env:X_URL|d") == "https://x" and resolve("env:NOPE_URL|d") == "d" and resolve(3) == 3
    cands = load_candidates(mini_root / "candidates.yaml")
    save_candidates(cands, mini_root / "c2.yaml", header="# hi\n")
    assert (mini_root / "c2.yaml").read_text().startswith("# hi") and load_candidates(mini_root / "c2.yaml") == cands
    clients = build_clients(cands, load_config(), {"default": "live", "C5": "replay"})
    assert type(clients["C1"]).__name__ == "GroqClient" and clients["C1"] is clients["C2"]
    assert type(clients["C5"]).__name__ == "ReplayOnlyClient" and type(clients["C6"]).__name__ == "LocalClient"
    with pytest.raises(ValueError):
        build_clients(cands, load_config(), {"default": "sim"}, world=None)
