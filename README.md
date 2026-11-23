# Mosaic: a heterogeneous recursion router

**Author:** Richard Abishai · **Contact:** [github.com/zacharias1219](https://github.com/zacharias1219)

Mosaic answers questions over very long support logs. It splits the work into many small model
calls, sends each call to the cheapest model and device that meets the quality bar, and tests
statistically whether the answers are no worse than a single model call over the log.

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
* **Evaluation is paired:** five pipelines on the same 720 questions, a paired bootstrap, McNemar and Holm,
  and non-inferiority verdicts whose wording was fixed in advance.

The full results are in **[report/report.md](report/report.md)**, and the reasoning behind each choice is in
**[DECISIONS.md](DECISIONS.md)**.

## Results

Three question sets appear below; each number says which one it comes from:
**live slice** (20 aggregate and pairwise questions at 128k, real Cerebras calls), **primary set** (the 240
aggregate and pairwise questions at 128k and 512k, simulated), and **all 720** (every class and size, simulated).

### Live, on Cerebras: 20 aggregate and pairwise questions at 128k

| pipeline | accuracy | aggregate (10) | pairwise (10) | cost per correct |
|---|---|---|---|---|
| P0: one direct call, corpus trimmed to the window | 5% (1/20) | 1/10 | 0/10 | $0.74 |
| P0f: filter in code, then one direct call | 5% (1/20) | 1/10 | 0/10 | $0.44 |
| P3: recursion with the learned router | **60% (12/20)** | 2/10 | 10/10 | $0.06 |

* P3 vs P0 and P3 vs P0f: **+55 points**, one-sided 95% CI [+30, +75], McNemar p = 0.003, superior.
* **The gain comes from splitting the work, not from filtering.** Filtering in code and then making one direct
  call (P0f) scored exactly what P0 scored, right on the same single question. The filter cut 1,130 tickets to 15–373
  for the aggregate questions, but a single direct call still miscounted on 9 of 10, even over 15–24 tickets.
* **The simulator was optimistic, and the live slice caught it.** Before seeing these questions it predicted 80% for P3
  (live: 60%). Six of P3's eight live misses were tone counts, where the simulator's error rate had rested on 3
  observations; targeted live runs measured Qwen's tone error at about 5% per ticket and gpt-oss-120b's at 15%. Part of
  that is the data: every ticket reports a problem, and a "positive" one differs only by its closing line.
* Live spend for all the Cerebras work: about $1.06 at list price.

### Live device study: 12 questions at 8k on every device

| weights | device | accuracy | map/direct call p50 | decode tok/s |
|---|---|---|---|---|
| Qwen 3.8 27B | Groq LPU | 91.7% | 1.6 s | 391* |
| Qwen 3.8 27B | Cerebras WSE | 91.7% | 0.6 s | 3,066 |
| gpt-oss-120b | Groq LPU | 83.3% | 2.1 s | 477 |
| gpt-oss-120b | Cerebras WSE | 66.7% | 0.8 s | 2,194 |
| gpt-oss-20b | Groq LPU | 83.3% | 1.3 s | 936 |
| Qwen 2.5 1.5B, 4-bit | local CPU (4 cores) | 41.7% | 103 s | 5 |

\*Groq's Qwen returns its JSON answer in one buffered chunk, so its decode speed is fitted rather than streamed.
Accuracy over 12 questions is anecdotal; the timings rest on every call. Same weights didn't always mean same
answers: Qwen matched exactly across devices, while gpt-oss-120b on Cerebras missed two questions it got right on
Groq. On Groq's free tier, waiting for the token budget (25–62 s per call) dominated latency.

### Simulated: the primary set (240 questions)

The full evaluation is simulated (run live it would take about 15 hours of Cerebras quota per pipeline), with error
rates and latencies fitted from the live calls above.

| pipeline | accuracy | cost per correct | p95 latency |
|---|---|---|---|
| P0 one direct call, corpus trimmed to the window | 2.9% | $1.10 | 9 s |
| P0f filter in code, then one direct call | 25.0%† | $0.088† | 9 s |
| P1 recursion, one candidate (Qwen 27B on Cerebras) | 60.0% | $0.127 | 301 s |
| P2 recursion, rule router | 57.9% | $0.133 | 301 s |
| P3 recursion, learned router | 59.6% | $0.130 | 301 s |

†An upper bound. The simulator's accuracy curve for direct calls is an assumption, and live it is too kind to short
filtered contexts: on the live slice it gave P0f 35%, against 5% measured.

* **P0 is not "one model reading everything".** Real tokenizers count about 13% more than Mosaic's estimate, so P0
  had to trim the corpus on all 240 primary questions, 128k included: it sees the first 84% of a 128k log and the first
  21% of a 512k one. "Beats P0" means *beats one model reading as much of the log as fits*. P0f is the stronger
  baseline, and the live slice is the better test of it.
* **Recursion against the baseline:** P3 vs P0 is superior (+56.7 points, one-sided 95% CI lower bound +51.2) at 0.12×
  the cost per correct answer (M1, M2 met).
* **Where routing hurt.** On the primary set, routing adds nothing measurable. P2 vs P1 is *inconclusive* (−2.1
  points, CI [−5.4, +1.2]) and P3 vs P2 is non-inferior (+1.7). Neither router beats fixed P1, which sends every map
  call to Qwen 27B, the cheapest and the most accurate mapper on measured data. In one cell routing clearly hurt:
  **on pairwise questions at 128k both routers score 76.7% against P1's 91.7%, 15 points worse** (P2 vs P1 is
  "inferior" on the cell's unadjusted CI; Holm-adjusted McNemar p = 0.047). Both sent 9% of map chunks to gpt-oss-120b,
  which got 76% of them right against Qwen's 97%, and the median latency stayed at 61.5 s: no gain bought the loss.
  P2 did it because Qwen's token budget was full and the rule table rates gpt-oss-120b above the bar; P3 did it
  because its gpt-oss-120b predictor, the miscalibrated one, ranked it above Qwen on those chunks.
* **Latency is not met (M3).** A full-corpus 512k question needs several minutes of Cerebras' 150k tokens a minute,
  however it is routed. P0 is fast because it reads only part of the log.
* **Calibration is partly there (M5 not met).** The Qwen (C5) mapper predictor is good (Brier 0.043 vs 0.067 for a
  constant, worst reliability bin 2 points off); gpt-oss-120b's (C4) worst bin is 12.4 points off and the CPU's 13.1,
  above the 10-point bar.
* **The placeholder price.** Qwen 3.8 27B on Cerebras has no list price we could confirm, so it is priced at a
  placeholder ($0.30 in / $0.60 out per million tokens), and it does almost all the work. The 0.12× ratio still holds
  because both pipelines are priced on that same model, so a common change in its price cancels. The report reprices
  the recorded token counts under five other prices, including lopsided input/output splits: P3/P0 stays between
  0.10× and 0.19×, inside the PRD's 1/3 target every time.
* Over all 720 questions (every class and size): P0 35.4%, P0f 58.2%†, P1 80.0%, P2 79.6%, P3 80.0%.

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

To re-run the whole pipeline offline (simulator calibration, profiling, router training, the five-pipeline
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
| `mosaic run` | run P0, P0f, P1, P2 or P3 on a question set; `--live`, `--fixed`, `--explore`, `--budget-ms`, `--no-cache`, `--dry-run` |
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
  on all six devices, and it is checked against a live Cerebras validation slice, which found it optimistic twice
  (tone questions, and direct calls on filtered contexts). The report labels every number as live or simulated.
* Per-cell verdicts in the report use unadjusted 95% intervals and are labelled that way; only the pooled primary
  analysis, fixed in advance, is a single confirmatory test.
* Mosaic is an independent reconstruction from public material. It makes no claim about Callosum's internals.
