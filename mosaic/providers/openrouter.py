"""Optional GPU row via OpenRouter. Never required."""
from mosaic.providers.base import OpenAICompatClient


class OpenRouterClient(OpenAICompatClient):
    key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"
