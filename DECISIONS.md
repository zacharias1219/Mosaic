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
**Evidence (current runs, `router/model.json`; held-out profiling data):**

| predictor | Brier | constant | beats constant | worst bin |
|---|---|---|---|---|
| mapper C5 (Qwen 27B, Cerebras) | 0.0432 | 0.0668 | yes (isotonic applied) | 2.1 pts |
| mapper C4 (gpt-oss-120b, Cerebras) | 0.1662 | 0.1937 | yes | 12.4 pts |
| mapper C6 (CPU) | 0.0088 | 0.0128 | yes | 13.1 pts |
| direct C6 | 0.1700 | 0.1760 | yes | 10.1 pts |
| direct C4, direct C5 | 0.0051, 0.0008 | the same | no (a tie: base rates ~0.98) | 7.1, 2.6 pts |

All three mapper predictors beat the base rate, and C5's (the one that routes almost all the work) is well
calibrated. M5 is still not met: C6's mapper bin is 13.1 points off and C4's 12.4, above the 10-point bar.
History: the first version, calibrated from Groq calls only, did *not* beat the constant for C4 and C5 (Brier 0.1430 vs
0.1424 and 0.0445 vs 0.0443). That is because the per-field error rates were then all small and similar, so whether a chunk came
back right was close to noise. The live Cerebras data (D16) showed tone errors at 5–15% per ticket against category
errors near 0.1–1%, which makes the schema features predictive.
**What it buys the router:** little. Over all 720 questions P3 scores 80.0%, the same as fixed P1 (80.0%) and just
above the rule router P2 (79.6%); on the 240-question primary set it is 59.6% vs P1's 60.0% and P2's 57.9% (P3 vs P2
non-inferior). Where it does act, it can hurt: on pairwise questions at 128k it sends 9% of map chunks to gpt-oss-120b
when neither model clears the bar, because the C4 predictor (the 12-point bin) ranks C4 above Qwen there. Those chunks
are right 76% of the time against Qwen's 97%, and the cell drops from 91.7% to 76.7% with no latency gain. The
honest position remains the PRD's gate 5: the rule router is as good, and per-template difficulty is the feature to
learn next. The test suite also checks that on pure-noise labels the report says the model loses to the base rate
rather than hiding it.

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

## D9. A deterministic simulator, calibrated from live calls, for the full evaluation
**Chosen:** `providers/sim.py` reads the ground truth and corrupts it at per-field rates that `mosaic calibrate`
fits from live, oracle-labelled map calls. It uses a Beta prior with 50 pseudo-observations so thin data stays near
the prior, dedupes identical calls, and skips category labels from tasks that restrict the label set. Same weights
share one error model (C4 ← C3, C5 ← C2). Latency is simulated in virtual time from per-candidate priors.
**Alternatives:** run the full evaluation live: about 15 hours of Cerebras quota per pipeline (150k tokens a
minute), and impossible on Groq's free plan (200k tokens a day per model). Cerebras was unreachable at first, so the
first calibration used Groq only; with a key, the device study added both Cerebras rows and the fit now uses all six
live devices (identical calls count once per device), plus a 20-question live validation slice (report § Live
validation).
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
**Multiplicity:** the pooled primary comparison is the single test fixed in advance, so its CI is not adjusted. The
per-cell rows are exploratory: their McNemar p-values are Holm-adjusted across cells, but their CIs and verdicts are
unadjusted 95% intervals, and the report and `mosaic eval` label those verdicts "(unadjusted)" rather than imply a
correction they don't carry.
**Evidence:** report § Sample size states the discordance actually observed and what a 2-point margin would cost.

## D13. Every number from records; every record replayable
**Chosen:** run files (gzipped JSONL, fixed gzip header) hold every span. The report reads only those files, uses
fixed SVG hash salts and no dates, and the live response cache is committed, so offline replays reproduce live runs
(`scripts/verify_repro.py`, M6). A `ReplayOnlyClient` makes any cache miss an error during offline replay, so a
replay can never silently call the network.
**Bug this caught:** in live mode, a call after a cached (replayed) call started on the real clock, before the
replayed call's logical end, which understated question latency. Calls now start at `max(logical time, admission)`.

## D14. Token counts carry a 1.2× margin against real tokenizers
**Chosen:** the proxy tokenizer stays (deterministic, provider-free), but every window check and the direct path's
truncation budget multiply it by 1.2, and Cerebras rows admit 85% of their published per-minute limits.
**Evidence:** the first live P0 call at 128k failed with HTTP 400 `context_length_exceeded`: a prompt the proxy
counted as 129.5k tokens was 146.7k to Cerebras' Qwen tokenizer (1.13×). Groq's 8k per-call cap had hidden this,
because no Groq call came near a window. With the margin, the trimmed corpus fits (one call, 10 s, $0.037).

## D15. Rate limits steer choices only among candidates that clear the bar
**Chosen:** the capacity step (skip a candidate that would queue while another has headroom) applies only within
the set predicted to clear τ. If every good candidate is queued, the call waits rather than going to an idle one
below the bar.
**Evidence:** after Cerebras' limits were tightened to 85%, pairwise questions at 128k+ saturated C5's token budget.
The old capacity step then left only the idle CPU model, which won as "highest predicted quality" at about 2%, and
46 questions turned wrong (P2 fell from 85.8% to 80.6% over all 720 questions). A regression test covers it
(`test_queue_never_pushes_a_call_below_the_bar`).

## D16. Validate the simulator live, and check every routing gain's evidence
**Chosen:** a live slice on the real main pool (P0 vs P3, 20 questions at 128k, about $0.90; P0f added later for $0.05 of new calls), compared per question
with what the simulator had predicted before seeing it (`precal-*-slice`); then recalibrate on it.
**Evidence:** the simulator was optimistic: on the 20 live-slice questions it predicted 80% for P3, and live gave 60%. Six of P3's eight live misses were tone
counts, and the tone error rate had rested on 3 observations. After recalibrating, the learned router appeared to
beat the rules (82.2% vs 79.6% over all 720 questions), but 22 of its 31 wins came from sending tone chunks to gpt-oss-120b, whose tone
rate rested on 6 observations. A targeted live run on those tone chunks (0/7 for C4, 1/7 for C5) measured C4's tone
error at 15% per ticket, 3× Qwen's, and the gain vanished (all 720 questions: P3 80.0% = P1 80.0%; 240-question
primary set: P3 59.6% vs P1 60.0%). A learned router will find whichever
parameter has the least evidence; every gain it shows needs that evidence checked.
**Data finding:** tone labels are ambiguous by construction. Every ticket reports a problem, and "positive" differs
only by a polite closing line, so the tone-count template measures label ambiguity as much as model quality.

## D17. Offline replays follow recorded live routing
**Chosen:** spans record a request ID (hash of the messages). Replaying a live run pins each request to the
candidate that served it live, because live routing depended on wall-clock quota state the replay can't recreate.
**Evidence:** replaying `live-P3-128k` re-decided three chunks of one question. Other questions had sent the same
chunks to the other candidate, so both answers were cached and nothing failed, but the cost differed by 2%. That run
predates request IDs and stays 19/20 exact in M6; later live runs replay exactly (test
`test_offline_replay_follows_recorded_live_routing`).

## D18. A stronger baseline: filter in code, then one direct call (P0f)
**Why:** P0 had to trim the corpus on all 240 primary questions, 128k included (real tokenizers count ~13% more than
the proxy, D14), so "beats one model reading everything" overstated it: P0 is *one model reading as much of the log as
fits*. The obvious objection is that recursion's gain might come from filtering in code, not from splitting the work.
**Chosen:** P0f uses the same shared plan as P1–P3, applies its header filter in code, and then makes a single direct
call on C5 over what is left. P0 → P0f isolates filtering; P0f → P1 isolates splitting into per-ticket extraction
plus reducing in code.
**Evidence (live, 20 questions at 128k):** P0f 5%, P0 5% (right on the same single question), P3 60%. On the 10
aggregate questions the filter cut 1,130 tickets to 15–373, and the one direct call still got 9 wrong, even over
15–24 tickets. The 10 pairwise plans have no header filter, so P0f read the same trimmed corpus and got 0/10, where
P3 got 10/10. (Those 10 pairwise calls are byte-identical requests to P0's, so they replayed from the cache: the
two baselines necessarily give the same answer there.) Live, the whole gain is splitting.
**The simulator disagrees, and live wins:** simulated P0f scores 25.0% on the primary set (P0 2.9%, P1 60.0%) and
35% on the live-slice questions. Its accuracy curve for direct calls is an assumption, and it is too kind to short
filtered contexts. The report flags the simulated P0f rows as an upper bound for filtering.
