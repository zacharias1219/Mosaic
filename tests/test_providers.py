"""9.4 Provider clients and 9.5 rate limits."""
import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest

from mosaic.providers.base import CallTooLarge, OpenAICompatClient, ProviderTimeout, RateLimited
from mosaic.providers.cerebras import CerebrasClient
from mosaic.providers.groq import GroqClient
from mosaic.providers.local import LocalClient, cpu_cost
from mosaic.providers.openrouter import OpenRouterClient
from mosaic.providers.ratelimit import Limiter, QuotaExhausted, VirtualClock
from mosaic.types import Candidate

FIX = Path(__file__).parent / "fixtures"
MSG = [{"role": "user", "content": "Return JSON"}]


def cand(**kw):
    base = dict(id="X", provider="groq", device="d", model="m", roles=["mapper"], context_window=8192)
    return Candidate(**{**base, **kw})


def sse_transport(body: bytes, status=200, headers=None, record=None):
    def handler(req: httpx.Request):
        if record is not None:
            record.append(req)
        return httpx.Response(status, content=body, headers=headers or {"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("cls", [GroqClient, CerebrasClient, LocalClient, OpenRouterClient])
async def test_recorded_fixture_round_trips_without_network(cls):
    body = (FIX / "groq_stream.sse").read_bytes()
    seen = []
    client = cls("https://example.invalid/v1", transport=sse_transport(body, record=seen))
    c = await client.complete(cand(reasoning_effort="low"), MSG, max_tokens=100)
    assert json.loads(c.text) == {"rows": []}
    assert c.tokens_in == 102 and c.tokens_out == 25 and c.reasoning_tokens == 12
    assert c.ttft_ms <= c.total_ms
    sent = json.loads(seen[0].content)
    assert sent["stream"] is True and sent["response_format"] == {"type": "json_object"}


async def test_local_fixture_round_trips():
    body = (FIX / "local_stream.sse").read_bytes()
    c = await LocalClient("http://x/v1", transport=sse_transport(body)).complete(cand(provider="local"), MSG)
    assert c.text and c.tokens_in > 0


async def test_ttft_measured_from_send_to_first_token():
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b'data: {"choices":[{"delta":{"content":"{\\"a\\""}}]}\n\n'
            await asyncio.sleep(0.10)
            yield b'data: {"choices":[{"delta":{"content":": 1}"}}],"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n'
            yield b"data: [DONE]\n\n"

    t = httpx.MockTransport(lambda r: httpx.Response(200, stream=Slow()))
    c = await OpenAICompatClient("http://x/v1", transport=t).complete(cand(), MSG)
    assert 40 <= c.ttft_ms <= 120 and c.total_ms >= c.ttft_ms + 90
    assert c.text == '{"a": 1}'


async def test_429_backoff_with_jitter_respects_retry_after():
    calls, sleeps = [], []
    body = (FIX / "groq_stream.sse").read_bytes()

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, content=b"slow down")
        return httpx.Response(200, content=body)

    async def fake_sleep(s):
        sleeps.append(s)

    client = GroqClient("http://x/v1", transport=httpx.MockTransport(handler), sleep=fake_sleep)
    c = await client.complete(cand(), MSG)
    assert c.retries == 2 and len(calls) == 3
    assert all(2.0 <= s <= 2.5 for s in sleeps)  # never below retry-after; jitter on top
    assert len(set(sleeps)) == 2  # jittered


async def test_429_gives_up_after_max_retries():
    async def nosleep(s):
        pass
    t = httpx.MockTransport(lambda r: httpx.Response(429, content=b""))
    with pytest.raises(RateLimited):
        await GroqClient("http://x/v1", transport=t, sleep=nosleep, max_retries=2).complete(cand(), MSG)


async def test_timeout_raises_typed_error():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)
    with pytest.raises(ProviderTimeout):
        await GroqClient("http://x/v1", transport=httpx.MockTransport(handler)).complete(cand(), MSG)


async def test_groq_cap_refuses_before_sending():
    seen = []
    client = GroqClient("http://x/v1", transport=sse_transport(b"", record=seen))
    big = [{"role": "user", "content": "x" * 4 * 6500}]
    with pytest.raises(CallTooLarge):
        await client.complete(cand(max_call_tokens=7000), big, max_tokens=1000)
    assert not seen and not client.sent_estimates


def test_token_bucket_never_exceeds_limits_in_60s_burst():
    lim = Limiter(rpm=30, tpm=8000, max_concurrency=4, virtual=True)
    starts = []
    for i in range(200):  # a burst of 200 calls issued at once
        s = lim.reserve(500 + (i % 7) * 100, at=0.0)
        lim.finish(s, s + 300)
        starts.append((s, 500 + (i % 7) * 100))
    starts.sort()
    for t0, _ in starts:
        win = [(s, k) for s, k in starts if t0 <= s < t0 + 60_000]
        assert len(win) <= 30 and sum(k for _, k in win) <= 8000
    # concurrency never above 4
    for t0, _ in starts:
        assert sum(1 for s, e in lim.intervals if s <= t0 < e) <= 4


def test_limiter_feasibility_and_daily_caps():
    lim = Limiter(rpm=5, tpm=30000, rpd=2, virtual=True)
    assert not lim.fits(32000)  # a 32k request can never fit C4's minute budget
    with pytest.raises(QuotaExhausted):
        lim.earliest(32000, 0)
    lim.reserve(100, 0)
    lim.reserve(100, 0)
    assert not lim.fits(100)
    # in virtual time an exhausted day pushes the next call to the next day's quota
    assert lim.wait_ms(100, 0) == 86_400_000
    real = Limiter(rpm=5, tpm=30000, rpd=1, virtual=False)
    real.record(0.0, 100)
    real.day_requests = 1
    assert real.wait_ms(100, 0) == float("inf")


def test_keys_from_env_and_never_logged(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "sk-secret-123")
    c = GroqClient("http://x/v1")
    assert c.headers()["Authorization"] == "Bearer sk-secret-123"
    from mosaic.providers.base import Completion
    from mosaic.providers.cache import ResponseCache, cache_key
    rc = ResponseCache(tmp_path / "c.sqlite")
    k = cache_key("groq", "m", MSG, {"max_tokens": 1})
    rc.put(k, "groq", "m", Completion(text="{}", tokens_in=1, tokens_out=1, ttft_ms=1, total_ms=2))
    blob = (tmp_path / "c.sqlite").read_bytes()
    assert b"sk-secret-123" not in blob
    assert "sk-secret-123" not in caplog.text


def test_cpu_cost_formula():
    # 28 W for 3600 s = 0.028 kWh at $0.10 = $0.0028, plus $0.05 amortisation
    assert abs(cpu_cost(3600, 28, 0.10, 0.05) - 0.0528) < 1e-9
