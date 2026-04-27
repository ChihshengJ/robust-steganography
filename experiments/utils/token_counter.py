"""Token and word counting utilities."""

from __future__ import annotations

import tiktoken

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder(encoding_name: str = "o200k_base") -> tiktoken.Encoding:
    """Lazily initialize and cache the tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(encoding_name)
    return _ENCODER


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens using tiktoken."""
    enc = _get_encoder(encoding_name)
    return len(enc.encode(text))


def count_words(text: str) -> int:
    """Whitespace-split word count."""
    return len(text.split())


def bits_per_token(num_bits: int, token_count: int) -> float:
    """Bits per token efficiency metric."""
    if token_count == 0:
        return 0.0
    return num_bits / token_count


def round_words(word_count: int, step: int = 50) -> int:
    """Round a word count to the nearest step (default 50)."""
    return round(word_count / step) * step
