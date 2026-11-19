"""Groq LPU (free plan). Every call must stay under 8k tokens in total, so the client refuses
anything it estimates above the candidate's max_call_tokens (7,000) before sending it."""
from mosaic.providers.base import OpenAICompatClient


class GroqClient(OpenAICompatClient):
    key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"
