import json
import os
import re
import time

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
    for attempt in range(3):
        try:
            kwargs = dict(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2**attempt)


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
