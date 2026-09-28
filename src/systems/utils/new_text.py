import json
import os
import random
import re
import time

import openai
import requests

from ..config.constants import STEGO_GEN_MODEL

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

REASONING_MODELS = {
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-codex",
}


def token_limit(client, n: int) -> dict:
    """The completion-length cap under the name the endpoint takes. OpenAI's
    reasoning-capable models reject ``max_tokens``; ``max_completion_tokens``
    means the same for its other models. Other OpenAI-compatible servers
    (OpenRouter, llama.cpp) take ``max_tokens``."""
    if "api.openai.com" in (client_base_url(client) or ""):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


# Longest single wait between retries: a per-minute rate limit resets within it.
MAX_RETRY_WAIT = 90.0

# Backoff jitter gets its own RNG: attacks seed the global `random` for
# sentence selection, and a retry must not shift that sequence.
_jitter = random.Random()


def retry_wait(error: Exception | None, attempt: int, base: float = 2.0) -> float:
    """Seconds to wait before retry ``attempt`` (1-based) after ``error``.

    A rate limit (429) waits until the provider says it resets: the
    Retry-After header, or OpenRouter's X-RateLimit-Reset (epoch ms, in the
    error body's metadata). Anything else backs off exponentially with jitter.
    Capped at MAX_RETRY_WAIT."""
    backoff = base * 2 ** (attempt - 1) * (1 + _jitter.random())
    if isinstance(error, openai.RateLimitError):
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(MAX_RETRY_WAIT, max(float(retry_after), 1.0))
            except ValueError:
                pass
        body = getattr(error, "body", None)
        meta = (body or {}).get("metadata", {}) if isinstance(body, dict) else {}
        reset = (meta.get("headers") or {}).get("X-RateLimit-Reset")
        if reset:
            try:
                wait = float(reset) / 1000 - time.time() + 1.0
                return min(MAX_RETRY_WAIT, max(wait, 1.0))
            except ValueError:
                pass
        return min(MAX_RETRY_WAIT, max(backoff, 10.0))
    return min(MAX_RETRY_WAIT, backoff)


LLM_ATTEMPTS = 6


class ContentFiltered(RuntimeError):
    """The provider refused the request (finish_reason content_filter); a
    retry cannot help, so llm raises it at once."""


def llm(
    client,
    model,
    prompt,
    system="You are a helpful assistant.",
    temperature=0,
    max_tokens=1000,
    top_p=0.7,
    extra_body: dict | None = None,
):
    for attempt in range(LLM_ATTEMPTS):
        try:
            kwargs = dict(
                model=model,
                **token_limit(client, max_tokens),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            # None leaves the provider default: OpenAI reasoning models reject
            # temperature/top_p unless reasoning is off.
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(**kwargs)
            if r.choices[0].finish_reason == "content_filter":
                raise ContentFiltered(f"{model} filtered the request (content_filter)")
            content = r.choices[0].message.content
            if not content:
                # e.g. a reasoning model spending the whole cap on reasoning
                raise RuntimeError(
                    f"empty completion from {model} "
                    f"(finish_reason={r.choices[0].finish_reason})"
                )
            return content.strip()
        except Exception as e:
            if isinstance(e, ContentFiltered) or attempt == LLM_ATTEMPTS - 1:
                raise
            wait = retry_wait(e, attempt + 1, base=1.0)
            print(f"  retry {attempt + 1} in {wait:.0f}s: {e}")
            time.sleep(wait)


def client_base_url(client) -> str | None:
    """The endpoint a client talks to, recorded so a stego text's provenance
    says where its models ran (e.g. the local GGUF server vs. a hosted API)."""
    url = getattr(client, "base_url", None)
    return str(url) if url is not None else None


def clean_response(text) -> str:
    # Regex to find the last full sentence ending with ., !, or ?
    match = re.search(r"([.!?])[^.!?]*$", text)
    if match:
        return text[: match.end()].strip()
    else:
        return text.strip()


def generate_response(
    prompt: str | list[str],
    system_prompt: str,
    max_length: int = 500,
    temperature: float = 0.7,
    top_p: float = 1.0,
    json_mode: bool = False,
    reasoning_effort: str
    | None = "minimal",  # "minimal", "low", "medium", "high", "xhigh"
    max_retries: int = 3,
) -> str:
    model = STEGO_GEN_MODEL
    if isinstance(prompt, list):
        prompt = "\n".join(prompt) + "\n"

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_length,
    }

    if model in REASONING_MODELS:
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
            if reasoning_effort == "none":
                payload["temperature"] = temperature
                payload["top_p"] = top_p
    else:
        payload["temperature"] = temperature
        payload["top_p"] = top_p

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {r.text}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}: {e}")
            time.sleep(2**attempt)
    return ""
