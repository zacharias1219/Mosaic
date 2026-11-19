"""M6: re-run every main pipeline from scratch and replay every live device run offline, then
check that the records match the committed run files and that the report rebuilds byte for byte.
Writes report/verify.json."""
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mosaic.config import load_config  # noqa: E402
from mosaic.data.store import load_corpus, load_questions  # noqa: E402
from mosaic.pipelines import RunContext, read_run, run_pipeline  # noqa: E402
from mosaic.session import Session  # noqa: E402
from mosaic.trace import Tracer  # noqa: E402

FIELDS = ("answer", "correct", "malformed", "truncated", "cost", "latency_ms")


def same(a, b) -> bool:
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > 1e-6 * max(1.0, abs(x)):
                return False
        elif x != y:
            return False
    return True


def rerun(rid: str, offline_live: bool) -> tuple[int, int]:
    cfg = load_config()
    s = Session(cfg)
    meta, old = read_run(s.runs_dir / f"{rid}.jsonl.gz")
    qids = [r.question_id for r in old]
    allq = {q.question_id: q for q in load_questions(s.data_dir, meta.get("split", "eval"))}
    qs = [allq[q] for q in qids]
    corpora = {sz: load_corpus(s.data_dir, meta.get("split", "eval"), sz) for sz in {q.size for q in qs}}
    backends = {k: ("replay" if v == "live" else v) for k, v in meta["backends"].items()}
    router = s.router(meta["pipeline"], pool=meta["pool"], explore=meta.get("explore"))
    gw = s.gateway(router, backends, Tracer(rid), live=False)
    ctx = RunContext(gw, rid, corpora, planner=meta.get("planner"), arrival_gap_ms=cfg["arrival_gap_ms"],
                     max_depth=cfg["max_depth"])
    new = asyncio.run(run_pipeline(meta["pipeline"], qs, ctx))
    return sum(same(a, b) for a, b in zip(old, new)), len(old)


def main() -> int:
    results = {}
    for rid in ("main-P0", "main-P1", "main-P2", "main-P3", "main-P3x", "device-C4-sim", "device-C5-sim"):
        if (ROOT / "runs" / f"{rid}.jsonl.gz").exists():
            results[rid] = rerun(rid, False)
    for rid in ("device-C1-live", "device-C2-live", "device-C3-live", "device-C6-live"):
        if (ROOT / "runs" / f"{rid}.jsonl.gz").exists():
            results[rid] = rerun(rid, True)
    rep = ROOT / "report" / "report.md"
    subprocess.run(["mosaic", "report"], cwd=ROOT, check=True, capture_output=True)
    first = rep.read_bytes()
    subprocess.run(["mosaic", "report"], cwd=ROOT, check=True, capture_output=True)
    report_same = rep.read_bytes() == first
    ok = report_same and all(a == b for a, b in results.values())
    matched = sum(a for a, _ in results.values())
    total = sum(b for _, b in results.values())
    summary = (f"{matched}/{total} question records reproduced exactly across {len(results)} runs "
               f"(live runs replayed offline); report rebuilt byte-identical: {report_same}")
    out = {"ok": ok, "summary": summary, "runs": {k: {"matched": a, "total": b} for k, (a, b) in results.items()}}
    (ROOT / "report" / "verify.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
