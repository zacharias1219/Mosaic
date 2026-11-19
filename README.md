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

From `report/report.md` (primary set: aggregate and pairwise questions at 128k and 512k, 240 paired questions).

| pipeline | accuracy | cost per correct | p95 latency |
|---|---|---|---|
| P0 one model reads everything (truncated to its window) | 3.8% | $1.02 | 11 s |
| P1 recursion, one candidate | 54.2% | $0.138 | 241 s |
| P2 recursion, rule router | 60.8% | $0.125 | 241 s |
| P3 recursion, learned router | 61.3% | $0.126 | 241 s |

* **Recursion is decisive, routing is marginal.** P3 vs P0 is superior (+57.5 points, one-sided 95% CI lower bound
  +51.7) at 0.12× the cost per correct answer (M1, M2 met). P3 vs P2 is non-inferior. Both routers mostly learn the
  same easy lesson: Qwen 27B on bulk maps, gpt-oss-120b on short critical-path calls, the CPU almost never.
* **Latency is not met (M3).** A full-corpus 512k question needs about four minutes of Cerebras' 150k tokens a minute,
  however it is routed. P0 is fast because it reads only a quarter of the corpus.
* **The learned predictor doesn't beat a constant for the two main mappers (M5 not met).** Live errors are
  systematic per phrasing, which call-level features can't see. This is reported as a negative result.
* **Live findings (Groq + local CPU, 12 questions each):** refund extraction was perfect on all three Groq models,
  paraphrases included. Qwen 3.8 27B got 100% of map calls right, against 94% for gpt-oss-120b and 89% for
  gpt-oss-20b. The real planner's one mistake was answering "how many customers…" with a list of IDs. Waiting for
  Groq's free-tier token budget (25–62 s per call on average) dominated latency, and a 1.5B model on 4 CPU cores was both slow (5 tok/s decode) and
  wrong on 94% of chunks.
* The main evaluation is simulated (Cerebras was unreachable from the build machine) with error rates fitted from
  the live calls. Every number in the report is labelled live or simulated.

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

* Cerebras was not reachable from the machine this was built on, so the main pool's Cerebras calls are simulated.
  The simulator is a deterministic test double whose error rates are fitted from live Groq and local-CPU calls
  with the same weights; its Cerebras speeds are priors. The report labels every number as live or simulated.
* Mosaic is an independent reconstruction from public material. It makes no claim about Callosum's internals.
