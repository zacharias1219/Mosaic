"""The only path to a provider: router -> rate limiter -> cache -> client (PRD 4).

Works on a real clock (live calls, real sleeps) or in virtual time (simulated calls), so the
same pipeline code produces both live and simulated runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from mosaic.providers.base import CallTooLarge, Completion, ProviderError
from mosaic.providers.cache import ResponseCache, cache_key
from mosaic.providers.local import cpu_cost
from mosaic.providers.ratelimit import Limiter, QuotaExhausted
from mosaic.routing.router import PoolState, Router
from mosaic.tokens import TOKEN_MARGIN, count_tokens
from mosaic.trace import Tracer
from mosaic.types import CallSpec, Candidate, Decision, Profile, Span


@dataclass
class CallResult:
    text: Optional[str]
    end: float
    span: Optional[Span]
    decision: Optional[Decision] = None
    too_large: bool = False


class SpendCapExceeded(Exception):
    pass


class Gateway:
    def __init__(self, candidates: dict[str, Candidate], clients: dict[str, Any], router: Router, tracer: Tracer,
                 clock, cache: Optional[ResponseCache] = None, profiles: Optional[dict[str, Profile]] = None,
                 limiters: Optional[dict[str, Limiter]] = None, local_cost: Optional[dict[str, float]] = None,
                 spend_cap_usd: Optional[float] = None, max_attempts: int = 4):
        self.candidates, self.clients, self.router, self.tracer = candidates, clients, router, tracer
        self.clock, self.cache = clock, cache
        self.profiles = profiles or {}
        self.limiters = limiters if limiters is not None else {
            c.id: Limiter(c.rpm, c.tpm, c.max_concurrency, c.rpd, c.tpd, virtual=clock.virtual, safety=c.limit_safety)
            for c in candidates.values()}
        self.local_cost = local_cost or {"watts": 28, "usd_per_kwh": 0.10, "amortisation_usd_per_hour": 0.05}
        self.spend_cap_usd = spend_cap_usd
        self.spent = 0.0
        self.max_attempts = max_attempts
        self.fallbacks = 0
        self.replay_hints: dict[tuple[str, str], list[str]] = {}  # (question, request) -> candidates used live

    def max_window(self, role: str) -> int:
        ws = [min(c.context_window, c.max_call_tokens or 10 ** 9) for c in self.candidates.values()
              if role in c.roles and (not self.router.pool or c.id in self.router.pool)
              and (self.router.mode != "fixed" or c.id == self.router.fixed)]
        return max(ws, default=8000)

    def price(self, c: Candidate, comp: Completion) -> float:
        if c.provider == "local":
            lc = self.local_cost
            return cpu_cost(comp.total_ms / 1000, c.watts or lc["watts"], lc["usd_per_kwh"], lc["amortisation_usd_per_hour"])
        return (comp.tokens_in * c.price_in + comp.tokens_out * c.price_out) / 1e6

    async def call(self, spec: CallSpec, messages: list[dict], max_tokens: int, at: float, qid: str,
                   parent_id: Optional[str] = None, exclude: Optional[set[str]] = None,
                   pin: Optional[str] = None) -> CallResult:
        exclude = set() if exclude is None else exclude
        request_id = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
        hint = self.replay_hints.get((qid, request_id))
        if hint and not pin:
            pin = hint.pop(0)  # replaying a live run: follow the choice it actually made
        retries = 0
        last_span = None
        failed: set[str] = set()   # candidates that failed during this call
        retried_failed = False
        for _ in range(self.max_attempts):
            state = PoolState(self.candidates, self.limiters, at, self.profiles, exclude, self.local_cost)
            dec = None
            if pin and pin not in exclude and pin in self.candidates:
                from mosaic.routing.router import predict_cost, predict_latency
                c = self.candidates[pin]
                lat = predict_latency(c, spec, self.profiles.get(pin))
                dec = Decision(candidate_id=pin, q_hat=0.0, l_hat=lat, c_hat=predict_cost(c, spec, lat, self.local_cost),
                               propensity=1.0, reason="pinned")
            else:
                dec = self.router.route(spec, state)
            if dec is None and failed and not retried_failed:
                # nothing else is feasible: give the failed candidates one more try (errors are often transient)
                retried_failed = True
                exclude -= failed
                dec = self.router.route(spec, PoolState(self.candidates, self.limiters, at, self.profiles, exclude,
                                                        self.local_cost))
            if dec is None:
                too_large = not exclude and all(
                    (spec.tokens_in_est + spec.tokens_out_est) * TOKEN_MARGIN > min(c.context_window, c.max_call_tokens or 10 ** 9)
                    for c in self.candidates.values() if spec.role in c.roles)
                return CallResult(None, at, last_span, None, too_large=too_large)
            c = self.candidates[dec.candidate_id]
            client = self.clients[c.id]
            need = spec.tokens_in_est + spec.tokens_out_est
            params = {"max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": c.reasoning_effort, "json": True}
            key = cache_key(c.provider, c.model, messages, params)
            cacheable = self.cache is not None and getattr(client, "cacheable", True)
            comp = self.cache.get(key) if cacheable else None
            if comp is None and cacheable and getattr(client, "replay_only", False):
                # offline replay: routing decisions made live depended on wall-clock quota state, so if the
                # router now picks differently, replay the candidate that actually served these messages live
                for alt in self.candidates.values():
                    if alt.id != c.id and alt.id not in exclude and getattr(self.clients.get(alt.id), "replay_only", False):
                        p_alt = {**params, "reasoning_effort": alt.reasoning_effort}
                        hit = self.cache.get(cache_key(alt.provider, alt.model, messages, p_alt))
                        if hit is not None:
                            c, client, comp, key = alt, self.clients[alt.id], hit, cache_key(alt.provider, alt.model, messages, p_alt)
                            dec = dec.model_copy(update={"candidate_id": alt.id, "reason": dec.reason + " (replayed live choice)"})
                            break
            lim = self.limiters[c.id]
            try:
                if self.spend_cap_usd is not None and comp is None and self.spent + dec.c_hat > self.spend_cap_usd:
                    raise SpendCapExceeded(f"spend cap ${self.spend_cap_usd} reached")
                if comp is not None:  # replay: original timings, including the queue wait seen live
                    start = at + comp.queue_ms
                    if self.clock.virtual:
                        lim.record(start, need)
                        lim.finish(start, start + comp.total_ms)
                elif self.clock.virtual:
                    start = lim.reserve(need, at)
                    comp = await client.complete(c, messages, max_tokens=max_tokens, json_mode=True)
                    lim.finish(start, start + comp.total_ms)
                else:
                    # logical time: a call cannot start before the replayed calls it follows end
                    start = max(at, await lim.acquire_real(need, self.clock))
                    try:
                        comp = await client.complete(c, messages, max_tokens=max_tokens, json_mode=True)
                    finally:
                        lim.release_real()
                if comp.tokens_out == 0 and comp.text:  # some servers omit usage when streaming
                    comp = comp.model_copy(update={"tokens_out": count_tokens(comp.text)})
                if not comp.replayed:
                    comp.queue_ms = start - at
                    if cacheable:
                        self.cache.put(key, c.provider, c.model, comp)
                end = start + comp.total_ms
                cost = self.price(c, comp)
                if not comp.replayed:
                    self.spent += cost
                span = self.tracer.add(
                    qid, spec.role, parent_id, start=at, end=end, candidate_id=c.id, tokens_in=comp.tokens_in,
                    tokens_out=comp.tokens_out, reasoning_tokens=comp.reasoning_tokens, ttft_ms=comp.ttft_ms,
                    queue_ms=start - at, total_ms=comp.total_ms, cost=cost, cache_hit=comp.replayed,
                    retries=retries + comp.retries, propensity=dec.propensity, q_hat=dec.q_hat, l_hat=dec.l_hat,
                    c_hat=dec.c_hat, explored=dec.explored, note=dec.reason, request_id=request_id,
                    features=spec.features)
                return CallResult(comp.text, end, span, dec)
            except SpendCapExceeded:
                raise
            except (ProviderError, CallTooLarge, QuotaExhausted) as e:
                elapsed = getattr(e, "elapsed_ms", 0.0)
                end = at + elapsed if self.clock.virtual else max(at, self.clock.now())
                last_span = self.tracer.add(qid, spec.role, parent_id, start=at, end=end, candidate_id=c.id, ok=False,
                                            retries=retries, propensity=dec.propensity, q_hat=dec.q_hat,
                                            note=f"failed: {type(e).__name__}: {str(e)[:120]}; falling back",
                                            features=spec.features)
                exclude.add(c.id)
                failed.add(c.id)
                retries += 1
                self.fallbacks += 1
                at = end
        return CallResult(None, at, last_span, None)
