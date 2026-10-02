"""Chat completions with retry, for the experiment stages that fan out API
calls over threads (normal generation, the LLM judge)."""

from __future__ import annotations

import time

import openai

from systems.utils.new_text import (
    DeadlineExceeded,
    create_completion,
    retry_wait,
    token_limit,
)

# Transient errors worth retrying. The SDK already retries these a couple of
# times internally; this outer loop rides out longer 429/503 bursts from
# providers under load.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    DeadlineExceeded,
)


class CompletionFailed(RuntimeError):
    """Every attempt at a completion failed."""


def chat(
    client,
    model: str,
    messages: list[dict],
    retries: int = 8,
    base_delay: float = 2.0,
    **kwargs,
) -> str:
    """One chat completion's text, retried on transient errors and empty
    completions (a rate limit waits for its reset, see retry_wait). Raises
    ``CompletionFailed`` once every attempt has failed."""
    if "max_tokens" in kwargs:
        kwargs |= token_limit(client, kwargs.pop("max_tokens"))
    last_error, last_exc = "no attempt made", None
    for attempt in range(retries):
        if attempt:
            time.sleep(retry_wait(last_exc, attempt, base_delay))
        try:
            response = create_completion(
                client, model=model, messages=messages, **kwargs
            )
        except _RETRYABLE as e:
            last_error, last_exc = repr(e), e
            continue
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            # The provider refused this input; asking again cannot help.
            raise CompletionFailed(f"{model} filtered the request (content_filter)")
        content = (choice.message.content or "").strip()
        if content:
            return content
        last_error, last_exc = "empty completion", None
    raise CompletionFailed(f"{retries} attempts failed; last error: {last_error}")
