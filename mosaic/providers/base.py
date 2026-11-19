"""One async interface over every OpenAI-compatible provider (PRD 9.4)."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from mosaic.tokens import count_messages, count_tokens
from mosaic.types import Candidate


class Completion(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    reasoning_tokens: int = 0
    ttft_ms: float
    total_ms: float
    status: int = 200
    model: str = ""
    replayed: bool = False
    queue_ms: float = 0.0
    retries: int = 0


class ProviderError(Exception):
    """Any failure after retries; the router catches it and falls back."""


class ProviderTimeout(ProviderError):
    pass


class RateLimited(ProviderError):
    pass


class CallTooLarge(ProviderError):
    """Refused before sending: the estimate exceeds the candidate's per-call cap."""


def parse_retry_after(headers: httpx.Headers) -> Optional[float]:
    v = headers.get("retry-after")
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


class OpenAICompatClient:
    """Streaming chat-completions client. TTFT is measured from request send to the first token
    (answer or reasoning) so hidden reasoning time is visible."""

    key_env = ""

    def __init__(self, base_url: str, key_env: str = "", timeout_s: float = 120.0, max_retries: int = 4,
                 transport: Optional[httpx.AsyncBaseTransport] = None, backoff_base_s: float = 1.0,
                 sleep=asyncio.sleep, seed: int = 0):
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env or self.key_env
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.transport = transport
        self.backoff_base_s = backoff_base_s
        self.sleep = sleep
        self.sent_estimates: list[int] = []  # every request actually sent (tests assert caps)
        self._rng = random.Random(seed)

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = os.environ.get(self.key_env, "") if self.key_env else ""
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def body(self, cand: Candidate, messages: list[dict], max_tokens: int, json_mode: bool,
             temperature: float) -> dict[str, Any]:
        b: dict[str, Any] = {"model": cand.model, "messages": messages, "max_tokens": max_tokens,
                             "temperature": temperature, "stream": True,
                             "stream_options": {"include_usage": True}, "seed": 42}
        if json_mode:
            b["response_format"] = {"type": "json_object"}
        if cand.reasoning_effort:
            b["reasoning_effort"] = cand.reasoning_effort
        return b

    def check_cap(self, cand: Candidate, messages: list[dict], max_tokens: int) -> int:
        est = count_messages(messages) + max_tokens
        if cand.max_call_tokens and est > cand.max_call_tokens:
            raise CallTooLarge(f"{cand.id}: estimated {est} tokens > cap {cand.max_call_tokens}")
        return est

    async def complete(self, cand: Candidate, messages: list[dict], max_tokens: int = 1024,
                       json_mode: bool = True, temperature: float = 0.0) -> Completion:
        est = self.check_cap(cand, messages, max_tokens)
        body = self.body(cand, messages, max_tokens, json_mode, temperature)
        retries = 0
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as http:
            while True:
                try:
                    self.sent_estimates.append(est)
                    comp = await self._stream(http, body, est)
                    comp.retries = retries
                    return comp
                except RateLimited as e:
                    retries += 1
                    if retries > self.max_retries:
                        raise
                    wait = e.args[1] if len(e.args) > 1 and e.args[1] else self.backoff_base_s * 2 ** (retries - 1)
                    await self.sleep(wait * (1 + 0.25 * self._rng.random()))  # jitter, never below retry-after
                except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                    raise ProviderTimeout(f"{cand.id}: timeout after {self.timeout_s}s") from e
                except httpx.HTTPError as e:
                    raise ProviderError(f"{cand.id}: {type(e).__name__}") from e

    async def _stream(self, http: httpx.AsyncClient, body: dict, est: int) -> Completion:
        t0 = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage: dict[str, Any] = {}
        async with http.stream("POST", f"{self.base_url}/chat/completions", json=body,
                               headers=self.headers()) as r:
            if r.status_code == 429:
                await r.aread()
                raise RateLimited(f"429 from {self.base_url}", parse_retry_after(r.headers))
            if r.status_code >= 400:
                text = (await r.aread()).decode(errors="replace")[:300]
                raise ProviderError(f"HTTP {r.status_code}: {text}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                if u:
                    usage = u
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if (delta.get("content") or delta.get("reasoning")) and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    if delta.get("content"):
                        parts.append(delta["content"])
        total = (time.perf_counter() - t0) * 1000
        details = usage.get("completion_tokens_details") or {}
        return Completion(
            text="".join(parts), tokens_in=int(usage.get("prompt_tokens", est)),
            tokens_out=int(usage.get("completion_tokens") or count_tokens("".join(parts))),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            ttft_ms=ttft if ttft is not None else total, total_ms=total, model=body["model"],
        )
