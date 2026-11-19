"""M7: count core lines (non-blank, non-comment, excluding docstring-only lines is not attempted)
and summarise coverage from report/coverage.json. Writes report/code_stats.json."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NON_CORE = {"cli.py", "report/report.py", "report/charts.py", "report/demo.py", "providers/sim.py"}


def sloc(p: Path) -> int:
    n, in_doc = 0, False
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if s.startswith(('"""', "'''")):
            if not (len(s) > 3 and s.endswith(('"""', "'''"))):
                in_doc = not in_doc
            continue
        if in_doc or not s or s.startswith("#"):
            continue
        n += 1
    return n


files = sorted((ROOT / "mosaic").rglob("*.py"))
rel = {str(f.relative_to(ROOT / "mosaic")): sloc(f) for f in files}
core = {k: v for k, v in rel.items() if k not in NON_CORE}
cov_path = ROOT / "report" / "coverage.json"
coverage = 0.0
per_file = {}
if cov_path.exists():
    cov = json.loads(cov_path.read_text())
    tot = cov_lines = 0
    for fname, d in cov["files"].items():
        k = str(Path(fname).resolve().relative_to(ROOT / "mosaic")) if "mosaic" in fname else fname
        if k in core:
            s = d["summary"]
            tot += s["num_statements"]
            cov_lines += s["covered_lines"]
            per_file[k] = round(s["percent_covered"], 1)
    coverage = 100 * cov_lines / max(1, tot)
tests = subprocess.run(["pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
n_tests = int(m.group(1)) if (m := re.search(r"(\d+)(?:/\d+)? tests? (?:collected|selected)", tests)) else 0
out = {"core_lines": sum(core.values()), "all_lines": sum(rel.values()), "coverage": round(coverage, 1),
       "tests": n_tests, "per_file_lines": rel, "per_file_coverage": per_file, "non_core": sorted(NON_CORE)}
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "code_stats.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(f"core lines {out['core_lines']} (all {out['all_lines']}), coverage on core {out['coverage']}%, tests {n_tests}")
