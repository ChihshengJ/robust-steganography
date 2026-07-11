"""DiscopSystem — Discop (S&P 2023) wrapped in the StegSystem interface.

In-house baseline, sibling to :class:`MeteorSystem`. Like Meteor it is a
token-level scheme over a local GPT-2 and is not expected to survive semantic
attacks; that failure is the comparison point against the paper's systems.

Decoding needs the same context (the ``seed`` string) and the same shared seed
(derived from ``key``), both re-established from the per-record ``system_state``
in Phase 4. Discop generates a variable number of tokens to embed a fixed
payload, so ``system_state`` also records ``error_encoded_length`` for the BER
truncation, exactly like the other systems.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..encoder import BypassEncoder, Encoder
from ..error_correction import ErrorCorrection, RepetitionCode
from ..steg_system import StegSystem
from . import _discop_backend as db


def _derive_seed(passphrase: str) -> int:
    """Map a passphrase to a 63-bit integer seed for Python's ``random``."""
    digest = hashlib.sha256(passphrase.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class DiscopSystem(StegSystem):
    def __init__(
        self,
        lm: db.DiscopLM,
        error_correction: ErrorCorrection | None = None,
        encoder: Encoder | None = None,
        key: str = "default",
        temp: float = 1.0,
        top_p: float = 0.92,
        max_length: int = 512,
    ) -> None:
        # Discop has no hash channel; bypass StegSystem.__init__ (needs a
        # HashFunction) and set only what the pipeline reads.
        self.lm = lm
        self.ecc = error_correction or RepetitionCode(1)
        self.encoder = encoder or BypassEncoder()
        self._seed_int = _derive_seed(key)
        self.temp = temp
        self.top_p = top_p
        self.max_length = max_length

        self._error_encoded_length: int | None = None
        self._context: str | None = None
        self._seed: str | None = None
        self._last_metadata: dict | None = None

    @property
    def error_encoded_length(self) -> int | None:
        return self._error_encoded_length

    @error_encoded_length.setter
    def error_encoded_length(self, value: int | None) -> None:
        self._error_encoded_length = value

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        m_bits = self.encoder.encode(data)
        m_encoded = self.ecc.encode(m_bits)
        self._error_encoded_length = len(m_encoded)
        self._context = seed
        self._seed = seed

        context_ids = self.lm.context_ids(seed)
        tokens, embedded = db.encode_discop(
            self.lm,
            list(m_encoded),
            context_ids,
            self._seed_int,
            temp=self.temp,
            top_p=self.top_p,
            max_length=self.max_length,
        )
        text = self.lm.tokenizer.decode(tokens)
        self._last_metadata = {
            "backend": "discop",
            "model": self.lm.model_name,
            "n_payload_bits": len(m_encoded),
            "n_bits_embedded": embedded,
            "n_tokens": len(tokens),
        }
        return text

    def recover_message(self, stego_text: str, **kwargs) -> Any:
        if self._error_encoded_length is None or self._context is None:
            raise ValueError(
                "DiscopSystem needs `error_encoded_length` and `context` set "
                "(run hide_message first, or restore them via system_state)."
            )
        context_ids = self.lm.context_ids(self._context)
        bits = db.decode_discop(
            self.lm,
            stego_text,
            context_ids,
            self._seed_int,
            temp=self.temp,
            top_p=self.top_p,
        )
        n = self._error_encoded_length
        raw = [int(b) for b in bits[:n]]
        if len(raw) < n:
            raw = raw + [0] * (n - len(raw))
        decoded = self.ecc.decode(raw, n)
        return self.encoder.decode(decoded)
