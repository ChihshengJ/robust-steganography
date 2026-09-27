"""Chat completions with retry, for the experiment stages that fan out API
calls over threads (normal generation, the LLM judge)."""

from __future__ import annotations

import random
import time

import openai

from systems.utils.new_text import token_limit

# Transient errors worth retrying. The SDK already retries these a couple of
# times internally; this outer loop rides out longer 429/503 bursts from
# providers under load.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class CompletionFailed(RuntimeError):
    """Every attempt at a completion failed."""


def chat(
    client,
    model: str,
    messages: list[dict],
    retries: int = 5,
    base_delay: float = 2.0,
    **kwargs,
) -> str:
    """One chat completion's text, retried with exponential backoff on
    transient errors and empty completions. Raises ``CompletionFailed`` once
    every attempt has failed."""
    if "max_tokens" in kwargs:
        kwargs |= token_limit(client, kwargs.pop("max_tokens"))
    last_error = "no attempt made"
    for attempt in range(retries):
        if attempt:
            time.sleep(base_delay * 2 ** (attempt - 1) * (1 + random.random()))
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except _RETRYABLE as e:
            last_error = repr(e)
            continue
        content = (response.choices[0].message.content or "").strip()
        if content:
            return content
        last_error = "empty completion"
    raise CompletionFailed(f"{retries} attempts failed; last error: {last_error}")
