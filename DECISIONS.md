# Decisions

One entry per design decision: what was chosen, the alternatives, and the evidence. "Report §" points at
sections of `report/report.md`; run IDs point at files in `runs/`.

## D1. JSON plans instead of model-written code
**Chosen:** the planner fills a strict pydantic schema (`mosaic/core/plan_schema.py`, `extra="forbid"`): strategy,
field filters, keywords with a `semantic` switch, chunk size, map task and schema, and a reducer from a fixed
library (`count`, `sum`, `set_union`, `pair_join`, `llm`). An invalid plan is rejected with a readable message and
the planner retries once with that message.
**Alternatives:** RLM-style model-written Python in a REPL (more general, needs a sandbox, and its failures are hard
to test); a fixed plan per question template (no planner at all, so nothing to learn from).
**Evidence:** all 12 live gpt-oss-120b plans validated on the first attempt (`plans` pass, `device-*-live`). The
schema needed one addition the PRD didn't list: `reduce_args` (which field to sum, what to count, the pair-join
categories and window). Without it a reducer can't be deterministic.

## D2. Code does arithmetic, models do semantics
**Chosen:** filtering on header fields, counting, summing and pair-joins run in Python. Models only extract
per-ticket facts.
**Alternatives:** an LLM aggregator over map outputs (still available as `reduce: llm`, never chosen by the live
planner).
**Evidence:** in the live device study, every refund flag and amount was extracted correctly (0 errors in 68
observations per model, paraphrases included), and the question-level misses came from plans and category
ambiguity, never from arithmetic. Recursion's error compounding is limited to one model step.

## D3. Judge a map call by what the reducer uses, not by raw field equality
**Chosen:** a map call is labelled correct when every ticket's facts, projected through the plan's reducer (the
where-clause hit, the summed amount, the pair-join category if it is one of the two that matter), match the ground
truth (`pipelines.project`).
**Alternative:** the first version compared every field with the oracle.
**Evidence:** the live planner wrote "classify each ticket as shipping, defect, or other" for a shipping/defect
pair question. The mapper then correctly called billing tickets "other", and raw equality scored 30 false errors
for gpt-oss-120b. With projection, the share of map calls fully correct went from 72% to 94% (C3), 67% to 89% (C1)
and 94% to 100% (C2). Those are the routing labels the predictor learns from, so this mattered.

## D4. Three router versions behind one decision rule
**Chosen:** R0 fixed, R1 a rule table by model tier and call shape, R2 a logistic regression per (role,
candidate). All three go through the same steps: feasibility (role, window, per-call cap, quota), capacity (skip a
candidate that would queue more than `max_queue_ms` while another has headroom), a latency-budget filter, the
quality bar τ, then min cost (off the critical path) or min latency (on it). If no candidate clears τ, the router
picks the highest predicted quality and flags the call.
**Alternatives:** a bandit learning online (needs far more traffic than a week allows); a single global model with
candidate one-hot (less inspectable).
**Evidence:** report § Router diagnostics: calls per candidate, flag share, held-out Brier per predictor.

## D5. Logistic regression, calibrated only when it helps
**Chosen:** standardised features, L2 logistic regression, a 70/30 split per (role, candidate), Brier compared with
a constant base-rate predictor, and isotonic scaling only when a held-out bin is more than 5 points off and
scaling lowers held-out Brier. With fewer than 30 rows the predictor is the base rate.
**Alternative:** gradient boosting (PRD open question 3). It isn't worth it at a few thousand labels per candidate,
and the coefficients are the explanation.
**Evidence (a negative result):** on held-out profiling data the C6 mapper and direct predictors beat the base
rate, but the C4 and C5 mapper predictors do not (Brier 0.1430 vs 0.1424 and 0.0445 vs 0.0443), and C5's worst
reliability bin is 21 points off, so M5 is not met. Adding products of features (tickets × fields) did not change
that beyond split noise. The live data explains why: map errors were systematic per phrasing, and whether a chunk
holds a "two-factor codes" ticket is invisible to call-level features. The learned router still matches the rule
router (82.2% vs 82.2% overall; non-inferior on the primary set) because the decisions that matter, keeping the
CPU off large chunks and putting C5 on bulk maps, are easy. As the PRD's gate 5 says, the honest position is: ship
the rule router, and treat per-template difficulty as the feature to learn next. The test suite also checks that
on pure-noise labels the report says the model loses to the base rate rather than hiding it.

## D6. Rate limits and quotas are routing inputs
**Chosen:** one limiter per candidate enforces requests and tokens per minute (sliding window), concurrency and
daily caps. The router reads its predicted queue wait. The same limiter runs on a real clock (sleeps) or in
virtual time (computes start times), so simulated runs queue exactly as live ones would.
**Alternative:** retry on 429 only. With Groq at 8k tokens a minute that becomes a retry storm.
**Evidence:** in the live device study, 78 distinct Groq calls were sent. The limiter kept C1 and C3 clear of
429s (1 retry in 56 calls), but Qwen on Groq (C2) drew 13 retries over 22 calls: the proxy tokenizer (4 characters
per token) undercounts against Groq's own token accounting. Every 429 was absorbed by `retry-after` plus jitter and
no call failed, but the right fix is to admit less, so Groq rows now carry `limit_safety: 0.75`. Mean queueing for
the free-tier token budget was 30–75 s per map call, the biggest latency term on the free plan (report § Device
study). In the simulation, C4's 5 requests a minute ration it to short critical-path calls.

## D7. Plans are shared across P1–P3
**Chosen:** the planner is pinned (`planner: C5` in the main evaluation, C3 in the device study), so P1, P2 and P3
see identical plans and chunks, and P1-vs-P2 isolates routing (test `test_p1_and_p2_differ_only_in_routing`).
**Alternative:** route the planner too. That confounds every comparison with plan differences, and the PRD requires
P1 and P2 to share plans and chunks.

## D8. Lookups take the direct path with a code filter
**Chosen:** `strategy: direct` plus a header filter that isolates the ticket, so a lookup is one small model call.
**Evidence:** report § Lookups (M4).

## D9. A deterministic simulator, calibrated from live calls, for what this machine can't reach
**Chosen:** `providers/sim.py` reads the ground truth and corrupts it at per-field rates that `mosaic calibrate`
fits from live, oracle-labelled map calls. It uses a Beta prior with 50 pseudo-observations so thin data stays near
the prior, dedupes identical calls, and skips category labels from tasks that restrict the label set. Same weights
share one error model (C4 ← C3, C5 ← C2). Latency is simulated in virtual time from per-candidate priors.
**Alternatives:** skip the main evaluation (no Cerebras key on the build machine); or run it on Groq (at 200k tokens
a day per model, ~1% of the evaluation).
**Evidence and caveats:** the live data showed errors are systematic per phrasing (gpt-oss reads "two-factor codes
rejected" as a defect and "renewal price went up" as other), while the simulator draws errors independently per
ticket. That understates correlated errors, so the report labels every simulated number and treats the main tables
as a test of the machinery under stated assumptions.

## D10. One simulated user with a minute of think time
**Chosen:** in virtual time, question *i+1* is asked 60 s after question *i* is answered. Bursts within a question
hit rate limits; backlog across questions doesn't turn into latency.
**Alternatives:** arrivals on a fixed clock (`max(i × 60 s, previous end)`) or back to back.
**Evidence:** the first version used the fixed clock. Once one slow question overran its minute, every later
question arrived the moment the previous one finished. P0 reads about 127k of C5's 150k tokens a minute, so each
of those calls then queued about 49 s for the previous one to leave the window, and P0's p50 latency read 60 s
instead of 11 s. That measured the arrival process, not the system.

## D11. Synthetic tickets with paraphrase and distractor banks
**Chosen:** labels are expressed only through phrasing. Category words never appear (a test checks this). At least
30% of refunds avoid the word "refund" (60% by construction), and 30% of non-refund tickets mention money or even the
word "refund" without being one ("I do not need a refund, I just want it fixed").
**Evidence:** keyword filters would miss most refunds; live models with `semantic: true` missed none. The ambiguity
that did hurt was in the category templates (D9), which is also the most realistic part.

## D12. Paired non-inferiority with a 5-point pooled margin
**Chosen:** as in the PRD: one-sided 95% paired bootstrap on accuracy differences, McNemar's exact test, Holm across
the per-cell tests, verdict wording fixed in advance. The primary analysis pools A and P at 128k and 512k (240
pairs).
**Evidence:** report § Sample size states the discordance actually observed and what a 2-point margin would cost.

## D13. Every number from records; every record replayable
**Chosen:** run files (gzipped JSONL, fixed gzip header) hold every span. The report reads only those files, uses
fixed SVG hash salts and no dates, and the live response cache is committed, so offline replays reproduce live runs
(`scripts/verify_repro.py`, M6). A `ReplayOnlyClient` makes any cache miss an error during offline replay, so a
replay can never silently call the network.
**Bug this caught:** in live mode, a call after a cached (replayed) call started on the real clock, before the
replayed call's logical end, which understated question latency. Calls now start at `max(logical time, admission)`.
