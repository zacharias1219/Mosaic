"""Deterministic simulated provider for offline runs and for candidates this machine cannot
reach (Cerebras). It is a test double, not a model: it reads the ground truth and corrupts it
at rates set per candidate in candidates.yaml (`sim:`), which `mosaic calibrate` fits from
live, oracle-labelled map calls. Every number produced through it is labelled "simulated".
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Optional

from mosaic.core.planner import reference_plan
from mosaic.core.reducers import reduce_rows, to_answer
from mosaic.data.oracle import oracle
from mosaic.providers.base import Completion, ProviderError
from mosaic.tokens import count_messages, count_tokens
from mosaic.types import Candidate, Question, Ticket

CATS = ["billing", "shipping", "defect", "account", "other"]
SENTS = ["positive", "neutral", "negative"]
DEFAULTS = {"err_easy": 0.003, "err_hard": 0.02, "drop": 0.002, "malformed": 0.005, "plan_err": 0.07,
            "direct_skill": 0.94, "agg_skill": 0.95, "error_rate": 0.003, "ctx_scale": 6000, "reasoning_tokens": 0,
            "T_L": 400_000, "T_A": 24_000, "T_P": 8_000, "decay_exp": 1.5}


KIND_PARAM = {"category": "err_category", "sentiment": "err_sentiment", "is_refund": "err_refund",
              "amount": "err_refund"}


class SimProviderError(ProviderError):
    def __init__(self, msg: str, elapsed_ms: float):
        super().__init__(msg)
        self.elapsed_ms = elapsed_ms


def field_kind(name: str, typ: str) -> str:
    n = name.lower()
    if typ == "category" or "categor" in n:
        return "category"
    if typ == "sentiment" or "sentiment" in n or "tone" in n:
        return "sentiment"
    if typ == "amount" or "amount" in n:
        return "amount"
    if typ == "bool" and ("refund" in n or "money" in n):
        return "is_refund"
    return "unknown"


def truth_of(t: Ticket, kind: str) -> Any:
    return {"category": t.category, "sentiment": t.sentiment, "amount": t.refund_amount,
            "is_refund": t.refund_amount is not None}.get(kind)


def hard(t: Ticket, kind: str) -> bool:
    low = t.text.lower()
    if kind in ("is_refund", "amount"):
        return (t.refund_amount is not None and "refund" not in low) or (
            t.refund_amount is None and ("refund" in low or "money back" in low or "$" in low))
    if kind == "category":
        return t.category in ("other", "billing") or "$" in low
    if kind == "sentiment":
        return t.sentiment == "neutral"
    return True


class SimWorld:
    def __init__(self) -> None:
        self.by_line: dict[str, Ticket] = {}
        self.questions: dict[str, Question] = {}

    def add_corpus(self, tickets: list[Ticket]) -> None:
        for t in tickets:
            self.by_line[t.line()] = t

    def add_questions(self, qs: list[Question]) -> None:
        for q in qs:
            self.questions[q.text] = q


def _after(s: str, marker: str, until: Optional[str] = None) -> str:
    i = s.find(marker)
    if i < 0:
        return ""
    s = s[i + len(marker):]
    if until is not None and until in s:
        s = s[: s.find(until)]
    return s


class SimClient:
    cacheable = False

    def __init__(self, world: SimWorld, seed: int = 42):
        self.world, self.seed = world, seed
        self._seen: dict[str, int] = {}  # transient errors are independent per attempt, not per message

    def _rng(self, cand: Candidate, messages: list[dict], salt: str = "") -> random.Random:
        h = hashlib.sha256(json.dumps([self.seed, cand.sim.get("weights", cand.model), cand.reasoning_effort,
                                       messages, salt], sort_keys=True).encode()).hexdigest()
        return random.Random(h)

    def p(self, cand: Candidate, k: str) -> float:
        return float(cand.sim.get(k, DEFAULTS[k]))

    def rate(self, cand: Candidate, kind: str, t: Ticket) -> float:
        """Per-ticket error rate for one field: the calibrated per-field rate when there is one,
        otherwise the prior easy/hard rate."""
        key = KIND_PARAM.get(kind)
        if key and key in cand.sim:
            return float(cand.sim[key])
        return self.p(cand, "err_hard" if hard(t, kind) else "err_easy")

    async def complete(self, cand: Candidate, messages: list[dict], max_tokens: int = 1024,
                       json_mode: bool = True, temperature: float = 0.0) -> Completion:
        system, user = messages[0]["content"], messages[-1]["content"]
        rng = self._rng(cand, messages)
        if user.startswith("That plan is invalid"):
            user = next(m["content"] for m in messages if m["role"] == "user")
        kind = system.split("\n", 1)[0]
        if kind == "Task type: map.":
            text = self._map(cand, user, rng)
        elif kind == "Task type: direct.":
            text = self._direct(cand, user, rng)
        elif kind == "Task type: plan.":
            text = self._plan(cand, user, rng)
        elif kind == "Task type: aggregate.":
            text = self._agg(cand, user, rng)
        else:
            text = '{"ok": true}'
        t_in = count_messages(messages)
        reasoning = int(self.p(cand, "reasoning_tokens") * math.exp(rng.gauss(0, 0.3)))
        t_out = count_tokens(text) + reasoning
        lat = random.Random(rng.random())  # timing noise independent of content decisions
        ttft = cand.ttft_ms * math.exp(lat.gauss(0, 0.25)) + 1000 * t_in / cand.prefill_tps
        total = ttft + 1000 * t_out / cand.tps_out * math.exp(lat.gauss(0, 0.1))
        k = hashlib.sha256(f"{cand.id}{json.dumps(messages, sort_keys=True)}".encode()).hexdigest()
        self._seen[k] = n = self._seen.get(k, 0) + 1
        if random.Random(f"{k}-{n}").random() < self.p(cand, "error_rate"):
            raise SimProviderError(f"simulated 5xx from {cand.id}", elapsed_ms=ttft)
        return Completion(text=text, tokens_in=t_in, tokens_out=t_out, reasoning_tokens=reasoning, ttft_ms=ttft,
                          total_ms=total, model=cand.model)

    # ---- behaviours ---------------------------------------------------------------------
    def _map(self, cand: Candidate, user: str, rng: random.Random) -> str:
        schema = json.loads(_after(user, "SCHEMA: ", "\n") or "{}")
        lines = [ln for ln in _after(user, "TICKETS:\n").split("\n") if ln.strip()]
        toks = sum(count_tokens(x) for x in lines)
        rot = 1 + toks / self.p(cand, "ctx_scale")
        if rng.random() < self.p(cand, "malformed") * (1 + len(lines) / 40):
            return '{"rows": [{"ticket_id": "' + (lines[0].split(" |")[0] if lines else "") + '", '  # truncated JSON
        rows = []
        for ln in lines:
            t = self.world.by_line.get(ln)
            tid = ln.split(" |")[0]
            if t is None or rng.random() < self.p(cand, "drop") * rot:
                continue
            row: dict[str, Any] = {"ticket_id": tid}
            for name, typ in schema.items():
                kind = field_kind(name, typ)
                v = truth_of(t, kind)
                err = self.rate(cand, kind, t) * rot
                if kind == "unknown" or rng.random() < err:
                    v = self._corrupt(kind, v, t, rng)
                row[name] = v
            rows.append(row)
        return json.dumps({"rows": rows})

    @staticmethod
    def _corrupt(kind: str, v: Any, t: Ticket, rng: random.Random) -> Any:
        if kind == "category":
            return rng.choice([c for c in CATS if c != v])
        if kind == "sentiment":
            return rng.choice([s for s in SENTS if s != v])
        if kind == "is_refund":
            return not v
        if kind == "amount":
            return None if v is not None else round(rng.choice([19.99, 44.99, 89.5, 129.99]), 2)
        return rng.choice([True, False, None])

    def _decay(self, cand: Candidate, cls: str, toks: int) -> float:
        T = self.p(cand, {"L": "T_L", "A": "T_A", "P": "T_P"}[cls])
        e = 1.0 if cls == "L" else self.p(cand, "decay_exp")
        return self.p(cand, "direct_skill") / (1 + (toks / T) ** e)

    def _direct(self, cand: Candidate, user: str, rng: random.Random) -> str:
        lines = [ln for ln in _after(user, "LOG:\n", "\n\nQUESTION: ").split("\n") if ln.strip()]
        qtext = _after(user, "\n\nQUESTION: ", "\n")
        key = next((k for k in ("customer_ids", "ticket_id", "amount", "count") if f'{{"{k}"' in user[-80:]), "count")
        q = self.world.questions.get(qtext)
        tickets = [self.world.by_line[ln] for ln in lines if ln in self.world.by_line]
        if rng.random() < self.p(cand, "malformed"):
            return f"The answer is probably {rng.randint(0, 50)}."
        if q is None:
            return json.dumps({key: None})
        try:
            truth = oracle(q, tickets)
        except StopIteration:  # the ticket was not in view
            truth = None
        toks = sum(count_tokens(x) for x in lines)
        if truth is not None and rng.random() < self._decay(cand, q.cls, toks):
            return json.dumps(to_answer(key, truth))
        return json.dumps(to_answer(key, self._wrong(key, truth, tickets, rng)))

    @staticmethod
    def _wrong(key: str, truth: Any, tickets: list[Ticket], rng: random.Random) -> Any:
        if key == "count":
            t = int(truth or 0)
            d = max(1, int(abs(t) * rng.uniform(0.05, 0.4)))
            return max(0, t - d) if rng.random() < 0.7 or t - d < 0 else t + d
        if key == "amount":
            t = float(truth or 0)
            return round(t * rng.uniform(0.55, 0.95), 2) if t else round(rng.uniform(10, 200), 2)
        if key == "customer_ids":
            ids = list(truth or [])
            if len(ids) > 1:
                return sorted(rng.sample(ids, max(1, int(len(ids) * rng.uniform(0.4, 0.8)))))
            return sorted({t.customer_id for t in rng.sample(tickets, min(2, len(tickets)))}) if tickets else []
        return rng.choice(tickets).ticket_id if tickets else None

    def _plan(self, cand: Candidate, user: str, rng: random.Random) -> str:
        qtext = _after(user, "QUESTION: ", "\n")
        q = self.world.questions.get(qtext)
        if q is None:
            return json.dumps({"strategy": "recurse", "map_task": "Classify each ticket.",
                               "map_schema": {"category": "category"}, "reduce": "count",
                               "reduce_args": {"where": {"category": "billing"}}})
        plan = reference_plan(q).model_dump()
        if rng.random() < self.p(cand, "plan_err"):
            r = rng.random()
            if q.cls == "L":
                plan["filter"]["fields"] = {}
            elif r < 0.55:
                plan["filter"]["semantic"] = False  # trusts keywords, misses paraphrases
                plan["filter"]["keywords"] = plan["filter"]["keywords"][:1] or ["refund"]
            elif r < 0.8:
                plan["strategy"] = "direct"
            else:
                plan["chunk_tokens"] = 16000
        if rng.random() < self.p(cand, "malformed"):
            return json.dumps(plan)[:-12]
        return json.dumps(plan)

    def _agg(self, cand: Candidate, user: str, rng: random.Random) -> str:
        rows = json.loads(_after(user, "FACTS:\n", "\n\nQUESTION: ") or "[]")
        qtext = _after(user, "\n\nQUESTION: ", "\n")
        q = self.world.questions.get(qtext)
        if q is None:
            return '{"count": 0}'
        plan = reference_plan(q)
        val = reduce_rows(rows, plan.reduce, plan.reduce_args) if plan.strategy == "recurse" else None
        key = plan.answer_key()
        if rng.random() > self.p(cand, "agg_skill"):
            val = self._wrong(key, val, [], rng)
        return json.dumps(to_answer(key, val))
