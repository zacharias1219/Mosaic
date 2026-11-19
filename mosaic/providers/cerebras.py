"""Cerebras wafer-scale. OpenAI-compatible; quotas live in candidates.yaml, not here."""
from mosaic.providers.base import OpenAICompatClient


class CerebrasClient(OpenAICompatClient):
    key_env = "CEREBRAS_API_KEY"
    default_base_url = "https://api.cerebras.ai/v1"

    def body(self, cand, messages, max_tokens, json_mode, temperature):
        b = super().body(cand, messages, max_tokens, json_mode, temperature)
        b.pop("stream_options", None)  # usage arrives in the final chunk without it
        return b
