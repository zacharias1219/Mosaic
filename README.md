# Mosaic: a heterogeneous recursion router

Mosaic answers questions over very long support logs. It splits the work into many small model
calls, sends each call to the cheapest model and device that meets the quality bar, and tests
statistically whether the answers are no worse than one big model reading everything.

```
question ─► planner ─► filter + chunk (code) ─► map × N (routed per chunk) ─► reduce (code) ─► answer
              │                                     │
              └────────── router: drop infeasible → predict quality → keep ≥ τ → min cost | min latency
                          (rate limits, context windows, latency budget are constraints)
```

* **Models do semantics, code does arithmetic.** The planner writes a restricted JSON plan (never code);
  filtering, counting, summing and pair-joins run in Python, so a model's only job is per-ticket extraction.
* **Every call goes through one gateway:** router → rate limiter → response cache → client. Groq, Cerebras,
  a local llama.cpp server and OpenRouter share one OpenAI-compatible streaming client that measures TTFT.
* **The router has three versions:** R0 fixed, R1 a hand-written rule table, R2 a calibrated logistic
  regression per (role, candidate) trained on oracle-labelled profiling calls, with 5% logged exploration
  for inverse-propensity estimates.
* **Evaluation is paired:** four pipelines on the same 720 questions, a paired bootstrap, McNemar and Holm,
  and non-inferiority verdicts whose wording was fixed in advance.

The full results are in **[report/report.md](report/report.md)**, and the reasoning behind each choice is in
**[DECISIONS.md](DECISIONS.md)**. Read the report's first section before its numbers: it says what was measured
live and what was simulated.

## Results at a glance

From `report/report.md` (primary set: aggregate and pairwise questions at 128k and 512k, 240 paired questions,
simulated main pool calibrated from live calls on all six devices).

| pipeline | accuracy | cost per correct | p95 latency |
|---|---|---|---|
| P0 one model reads everything (truncated to its window) | 2.9% | $1.10 | 9 s |
| P1 recursion, one candidate (Qwen 27B on Cerebras) | 60.0% | $0.127 | 301 s |
| P2 recursion, rule router | 57.9% | $0.133 | 301 s |
| P3 recursion, learned router | 59.6% | $0.130 | 301 s |

* **Recursion is decisive; routing isn't.** P3 vs P0 is superior (+56.7 points, one-sided 95% CI lower bound +51.2)
  at 0.12× the cost per correct answer (M1, M2 met). P3 vs P2 is non-inferior, and P3 ties P1: on measured data,
  Qwen 3.8 27B on Cerebras is both the cheapest and the most accurate mapper, so there is little to route around.
* **Live validation on Cerebras** (20 questions at 128k, real calls): P0 5% vs P3 60%, superior (p = 0.003).
  Out of sample, the simulator had predicted 80% for P3. The gap was tone questions, and two rounds of targeted live
  measurement fixed the simulator: Qwen's real tone error is 5× the prior, gpt-oss-120b's is 15× (report § Live
  validation).
* **Latency is not met (M3).** A full-corpus 512k question needs several minutes of Cerebras' 150k tokens a minute,
  however it is routed. P0 is fast only because it reads a quarter of the corpus.
* **Calibration is partly there (M5 not met).** The C5 mapper predictor is good (Brier 0.043 vs 0.067 constant,
  worst bin 2 points off); C4's worst bin is 13 points off.
* **Live device study, 12 questions per device:** Groq (gpt-oss-20b, Qwen 27B, gpt-oss-120b), Cerebras
  (gpt-oss-120b, Qwen 27B) and a 1.5B model on this machine's CPU. Refund extraction was perfect everywhere except
  the CPU. Same weights didn't always mean same answers: gpt-oss-120b on Cerebras missed two questions it got
  right on Groq, while Qwen matched exactly. The CPU was slow (5 tok/s) and wrong on 94% of chunks.
* Total live spend for the Cerebras work was about $1 at list price.

## Quick start (offline, no keys)

```bash
pip install -e '.[dev]'
```

```bash
make data && make report && make demo
```

`make data` regenerates the corpora deterministically, `make report` rebuilds `report/report.md` from the committed
run records, and `make demo` replays one aggregate question end to end from those records. None of these call a
provider, and all of them work without network access or keys.

To re-run the whole pipeline offline (simulator calibration, profiling, router training, the four-pipeline
evaluation, coverage, the reproducibility check and the report):

```bash
make all
```

## Live runs

Keys are read only from environment variables (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENROUTER_API_KEY`) and are
never written to logs or the cache. The local CPU candidate needs a llama.cpp server on port 8080:

```bash
pip install 'llama-cpp-python[server]'
```

```bash
python -m llama_cpp.server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --n_ctx 8192 --model_alias qwen2.5-1.5b-instruct-q4
```

Then the live device study (Groq free plan: about 10 minutes, well inside the 200k tokens a day per model):

```bash
make device
```

Any candidate can be served live in any run with `--live`, for example the Cerebras rows once a key is available:

```bash
mosaic run --pipeline P1 --sizes 8k --n-per-class 4 --fixed C4 --planner C3 --live C4,C3
```

Every paid command prints an estimate first, asks above `confirm_above_usd`, stops at `spend_cap_usd`, and has
`--dry-run`. Every live response is cached in `.mosaic/cache.sqlite`, so re-runs and `--resume` cost nothing.

## Commands

| Command | What it does |
|---|---|
| `mosaic gen` | corpora (8k–512k tokens) and question sets with exact answers computed in code |
| `mosaic candidates` | refresh `candidates.yaml` from each provider's `/models`, and warn on rate-limit header drift (`--check-limits`) |
| `mosaic profile` | per-candidate TTFT, prefill and decode speed, error rate, and oracle-labelled quality |
| `mosaic calibrate` | fit the simulator's error model from live, labelled map calls (offline replay) |
| `mosaic train-router` | fit and calibrate the quality predictor; prints Brier vs a constant predictor |
| `mosaic run` | run P0–P3 on a question set; `--live`, `--fixed`, `--explore`, `--budget-ms`, `--no-cache`, `--dry-run` |
| `mosaic eval` | paired per-cell and pooled non-inferiority verdicts between two runs (exit 3 if inconclusive) |
| `mosaic report` | `report/report.md` plus charts, from run records only |
| `mosaic demo` | one question end to end with each call's route, latency and cost, next to the baseline |

Exit codes: 0 success, 1 user error, 2 provider failure after retries, 3 evaluation inconclusive.

## Layout

```
mosaic/
  data/        gen.py  questions.py  oracle.py  store.py
  providers/   base.py  groq.py  cerebras.py  local.py  openrouter.py  ratelimit.py  cache.py  sim.py
  core/        planner.py  plan_schema.py  prompts.py  executor.py  reducers.py
  routing/     router.py  predictor.py  features.py  profiler.py
  eval/        scorer.py  stats.py  compare.py
  report/      charts.py  report.py  demo.py
  gateway.py  trace.py  pipelines.py  session.py  config.py  cli.py
tests/         one module per component, recorded provider fixtures, an opt-in live suite (pytest -m live)
runs/          committed run records (gzipped JSONL) behind every number in the report
profiles/ router/ report/   profiles, routing labels, the trained predictor, the report
```

`mosaic.yaml` holds thresholds, margins, budgets and the main pool. `candidates.yaml` holds every (model, device)
row: limits, prices, latency priors and the simulator's fitted error model.

## Honesty notes

* The full 720-question evaluation is simulated: run live, it would take about 15 hours of Cerebras quota per
  pipeline. The simulator is a deterministic test double whose error rates and latencies are fitted from live calls
  on all six devices, and it is checked against a live Cerebras validation slice. The report labels every number as
  live or simulated.
* Mosaic is an independent reconstruction from public material. It makes no claim about Callosum's internals.
