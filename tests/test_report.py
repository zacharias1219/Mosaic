"""9.14 Reporter."""
import re
import shutil
from pathlib import Path

import numpy as np

from mosaic.config import ROOT, load_config
from mosaic.pipelines import write_run
from mosaic.report.charts import non_dominated
from mosaic.report.report import MAIN, build_report
from mosaic.types import RunRecord, Span


def fake_runs(root: Path):
    rng = np.random.default_rng(0)
    acc = {"P0": 0.3, "P1": 0.85, "P2": 0.84, "P3": 0.86}
    cost = {"P0": 0.03, "P1": 0.01, "P2": 0.006, "P3": 0.005}
    for p, rid in MAIN.items():
        recs = []
        for cls in "LAP":
            for size in ("8k", "32k", "128k", "512k"):
                for i in range(20):
                    qid = f"q-{cls}-{size}-{i}"
                    sp = [Span(run_id=rid, question_id=qid, span_id=f"{qid}:1", role="question"),
                          Span(run_id=rid, question_id=qid, span_id=f"{qid}:2", parent_id=f"{qid}:1", role="mapper",
                               candidate_id="C5", cost=cost[p], correct=bool(rng.random() < 0.9), q_hat=0.9,
                               propensity=0.95, l_hat=500, c_hat=cost[p], total_ms=450)]
                    recs.append(RunRecord(run_id=rid, pipeline=p, question_id=qid, cls=cls, size=size,
                                          correct=bool(rng.random() < acc[p]), cost=cost[p],
                                          latency_ms=float(rng.uniform(500, 9000)), spans=sp,
                                          plan={"strategy": "direct" if cls == "L" else "recurse",
                                                "filter": {"fields": {"ticket_id": "T"}}}))
        write_run(root / "runs" / f"{rid}.jsonl.gz", {"run_id": rid, "git_commit": "abc", "data_hash": "d", "seed": 42}, recs)


def test_report_regenerates_byte_identically_and_traces_run_ids(tmp_path):
    shutil.copy(ROOT / "candidates.yaml", tmp_path / "candidates.yaml")
    fake_runs(tmp_path)
    cfg = load_config()
    p = build_report(tmp_path, cfg)
    first = p.read_text()
    svgs = {f.name: f.read_bytes() for f in (tmp_path / "report").glob("*.svg")}
    p2 = build_report(tmp_path, cfg)
    assert p2.read_text() == first
    assert {f.name: f.read_bytes() for f in (tmp_path / "report").glob("*.svg")} == svgs
    for rid in MAIN.values():
        assert rid in first
    assert "Not resolved by this sample" in first or "superior" in first
    assert re.search(r"Non-dominated on .*: .*P\d", first)


def test_pareto_marks_non_dominated_set():
    pts = [("P0", 0.03, 9000, 0.3), ("P1", 0.01, 4000, 0.85), ("P2", 0.006, 4500, 0.84), ("P3", 0.008, 5000, 0.80)]
    assert non_dominated(pts) == {"P1", "P2"}
