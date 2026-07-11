"""MeteorSystem — Meteor (CCS 2021) wrapped in the StegSystem interface.

An in-house baseline for the paper's robustness/detectability experiments. Unlike
the paper's semantic-channel systems, Meteor encodes at the token level via a
local causal LM, so it is *not* expected to survive paraphrase/translation
attacks — that contrast is the point of including it.

Decoding requires re-running the same LM from the same context, exactly the
determinism constraint the TopicQA/Story systems already satisfy with the local
llama.cpp server. Here the context is the ``seed`` string; it is saved in the
per-record ``system_state`` (as ``context``) so Phase 4 can decode a stego text
generated hours earlier. The symmetric key/nonce are fixed per system instance
(shared out-of-band), mirroring the LitReview keyed scheme.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..encoder import BypassEncoder, Encoder
from ..error_correction import ErrorCorrection, RepetitionCode
from ..steg_system import StegSystem
from . import _meteor_backend as mb


def _derive_key(passphrase: str) -> bytes:
    """Stretch a passphrase into Meteor's 64-byte DRBG key."""
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), b"meteor_salt_", 100000, dklen=64)


class MeteorSystem(StegSystem):
    def __init__(
        self,
        lm: mb.MeteorLM,
        error_correction: ErrorCorrection | None = None,
        encoder: Encoder | None = None,
        key: str = "default",
        nonce: bytes = b"\x01" * 64,
        temp: float = 0.95,
        precision: int = 32,
        topk: int = 50000,
    ) -> None:
        # Meteor has no PCA/hash channel, so we intentionally bypass
        # StegSystem.__init__ (which requires a HashFunction) and set only the
        # attributes the pipeline reads.
        self.lm = lm
        self.ecc = error_correction or RepetitionCode(1)
        self.encoder = encoder or BypassEncoder()
        self._key_bytes = _derive_key(key)
        self._nonce = nonce
        self.temp = temp
        self.precision = precision
        self.topk = topk

        self._error_encoded_length: int | None = None
        self._context: str | None = None  # the seed used at encode time
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

        context_tokens = self.lm.encode_context(seed)
        tokens = mb.encode_meteor(
            self.lm,
            list(m_encoded),
            context_tokens,
            temp=self.temp,
            precision=self.precision,
            topk=self.topk,
            input_key=self._key_bytes,
            input_nonce=self._nonce,
        )
        text = self.lm.enc.decode(tokens)
        self._last_metadata = {
            "backend": "meteor",
            "model": self.lm.model_name,
            "n_payload_bits": len(m_encoded),
            "n_tokens": len(tokens),
        }
        return text

    def recover_message(self, stego_text: str, **kwargs) -> Any:
        if self._error_encoded_length is None or self._context is None:
            raise ValueError(
                "MeteorSystem needs `error_encoded_length` and `context` set "
                "(run hide_message first, or restore them via system_state)."
            )
        context_tokens = self.lm.encode_context(self._context)
        bits = mb.decode_meteor(
            self.lm,
            stego_text,
            context_tokens,
            temp=self.temp,
            precision=self.precision,
            topk=self.topk,
            input_key=self._key_bytes,
            input_nonce=self._nonce,
        )
        n = self._error_encoded_length
        raw = [int(b) for b in bits[:n]]
        # Pad short recoveries (e.g. attacked text too short) so downstream BER
        # sees a fixed-length vector rather than treating it as a hard mismatch.
        if len(raw) < n:
            raw = raw + [0] * (n - len(raw))
        decoded = self.ecc.decode(raw, n)
        return self.encoder.decode(decoded)
