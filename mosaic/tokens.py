"""Fixed proxy tokenizer: one token per four characters, rounded up.

Deterministic and provider-independent. Real tokenizers differ by a few percent, which is
why every hard limit (Groq's 8k per minute) is enforced with a safety margin.
"""
import math

# Real tokenizers count more than the proxy on this text: Cerebras' Qwen tokenizer counted a 129.5k-proxy-token
# prompt as 146.7k tokens (1.13x). Window checks and truncation budgets use this margin.
TOKEN_MARGIN = 1.2


def count_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def count_messages(messages: list[dict]) -> int:
    return sum(count_tokens(m.get("content") or "") + 4 for m in messages)
