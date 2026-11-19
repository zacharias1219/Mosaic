"""Local CPU via a llama.cpp OpenAI-compatible server. Free, slow, unlimited; optional and
auto-detected. Cost is electricity plus amortisation (PRD 11)."""
from __future__ import annotations

import httpx

from mosaic.providers.base import OpenAICompatClient


class LocalClient(OpenAICompatClient):
    default_base_url = "http://localhost:8080/v1"

    def body(self, cand, messages, max_tokens, json_mode, temperature):
        b = super().body(cand, messages, max_tokens, json_mode, temperature)
        b.pop("seed", None)
        return b


def cpu_cost(seconds: float, watts: float, usd_per_kwh: float, amortisation_usd_per_hour: float) -> float:
    return watts * seconds / 3.6e6 * usd_per_kwh + seconds / 3600 * amortisation_usd_per_hour


def detect(base_url: str = LocalClient.default_base_url, timeout_s: float = 1.0) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
