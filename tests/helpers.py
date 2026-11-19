"""Small fakes shared by tests: a scripted client and a gateway factory."""
import asyncio

from mosaic.gateway import Gateway
from mosaic.providers.base import Completion, ProviderError
from mosaic.providers.ratelimit import RealClock, VirtualClock
from mosaic.routing.router import Router
from mosaic.trace import Tracer
from mosaic.types import Candidate


def cand(cid, **kw):
    base = dict(id=cid, provider="cerebras", device="d", model=f"model-{cid}", roles=["mapper", "planner", "direct",
                                                                                     "aggregator"],
                context_window=131072, price_in=0.3, price_out=0.6, rpm=1000, tpm=10 ** 7, max_concurrency=8,
                ttft_ms=200, prefill_tps=10000, tps_out=1000, sim={"tier": "mid"})
    base.update(kw)
    return Candidate(**base)


class ScriptedClient:
    """Returns texts from a function of (candidate, messages); optional real sleep and failures."""
    cacheable = False

    def __init__(self, fn, sleep_s=0.0, fail_ids=(), total_ms=100.0):
        self.fn, self.sleep_s, self.fail_ids, self.total_ms = fn, sleep_s, set(fail_ids), total_ms
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def complete(self, c, messages, max_tokens=100, json_mode=True, temperature=0.0):
        self.calls.append((c.id, messages))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.sleep_s:
                await asyncio.sleep(self.sleep_s(c) if callable(self.sleep_s) else self.sleep_s)
            if c.id in self.fail_ids:
                raise ProviderError(f"{c.id} is down")
            text = self.fn(c, messages)
            ms = self.sleep_s(c) * 1000 if callable(self.sleep_s) else (self.sleep_s * 1000 or self.total_ms)
            return Completion(text=text, tokens_in=100, tokens_out=20, ttft_ms=ms / 2, total_ms=ms, model=c.model)
        finally:
            self.active -= 1


def gateway(cands, client, mode="rules", fixed=None, real=False, **router_kw):
    cands = {c.id: c for c in cands}
    router = Router(mode, fixed=fixed, **router_kw)
    clock = RealClock() if real else VirtualClock()
    return Gateway(cands, {cid: client for cid in cands}, router, Tracer("t"), clock)
