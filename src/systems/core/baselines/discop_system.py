"""DiscopSystem — Discop (S&P 2023) wrapped in the StegSystem interface.

In-house baseline. A token-level scheme over a local GPT-2: the payload lives
in *which token gets sampled*, so it is not expected to survive a semantic
attack that rewrites the surface. That failure is the comparison point against
the paper's systems.

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
from ..error_correction import ERASURE, ErrorCorrection, RepetitionCode
from ..steg_system import StegSystem
from . import _discop_backend as db
from ._syncpool import derive_syncpool_key


def _derive_seed(passphrase: str) -> int:
    """Map a passphrase to a 63-bit integer seed for Python's ``random``."""
    digest = hashlib.sha256(passphrase.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _exhaustion_hint(stats: dict, n_tokens: int) -> str:
    """Explain *why* the budget ran out, from the encoder's step telemetry.

    Distinguishes a budget that is honestly too small from an entropy collapse,
    because only the first one is fixed by raising the budget.
    """
    last = stats.get("last_bit_step")
    singles = stats.get("n_singleton_steps")
    if last is None or singles is None or not n_tokens:
        return " Raise max_length or lower the repetition rate."
    dead_tail = n_tokens - 1 - last
    if dead_tail > 0.2 * n_tokens:
        return (
            f" The last bit went in at step {last}, leaving {dead_tail} tokens "
            f"({dead_tail / n_tokens:.0%}) that embedded nothing, and "
            f"{singles} step(s) had a single-candidate nucleus. That is an "
            f"entropy collapse — GPT-2 fell into a repetition loop and top_p "
            f"left the sampler no choice — not a budget shortfall. Raising "
            f"max_length will NOT help; the state is absorbing. Regenerate this "
            f"document (a different payload or key takes a different "
            f"trajectory), or reduce the length target so generations stay "
            f"short enough not to degenerate."
        )
    return (
        f" Bits were still going in at step {last}, so this is a genuine budget "
        f"shortfall: raise max_length or lower the repetition rate."
    )


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
        syncpool: bool = True,
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
        # SyncPool (Qi et al., arXiv:2403.17524) — Discop is the scheme they
        # validated it on. On by default because without it the clean *text*
        # channel is not exact: the sender must detokenize before transmitting,
        # and `token ids -> text -> token ids` is not the identity under greedy
        # BPE, so the receiver walks a different token sequence than the encoder
        # wrote and every later bit is chance — with no attacker present. Its
        # key is derived separately so Discop's own tree-walk RNG is untouched,
        # and `restore_system_state` overrides it from the record because a
        # SyncPool stream decoded without it yields chance.
        self.syncpool = syncpool
        self._syncpool_key = derive_syncpool_key(key)

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
        enc_stats: dict = {}
        if self.syncpool:
            tokens, embedded = db.encode_discop_syncpool(
                self.lm,
                list(m_encoded),
                context_ids,
                self._seed_int,
                temp=self.temp,
                top_p=self.top_p,
                max_length=self.max_length,
                syncpool_key=self._syncpool_key,
                stats=enc_stats,
            )
        else:
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
        if embedded < len(m_encoded):
            # Generation ran out of budget mid-payload. The trailing bits were
            # never written, so this record is unrecoverable before any attack
            # touches it — failing loudly beats emitting a record that looks
            # like a robustness failure but is a generation-budget failure.
            #
            # Two very different causes, and the advice differs, so say which.
            # A budget that is genuinely too small keeps embedding bits right up
            # to the cap. An entropy collapse — GPT-2 in a repetition loop, where
            # top_p leaves a single candidate, so the step embeds nothing and the
            # sampler cannot escape — stops embedding early and then burns every
            # remaining token for free. No budget rescues that one.
            raise ValueError(
                f"DiscopSystem: generation budget max_length={self.max_length} "
                f"exhausted after embedding {embedded}/{len(m_encoded)} payload "
                f"bits ({len(tokens)} tokens)."
                + _exhaustion_hint(enc_stats, len(tokens))
            )
        self._last_token_ids = list(tokens)
        self._last_metadata = {
            "backend": "discop",
            "model": self.lm.model_name,
            "n_payload_bits": len(m_encoded),
            "n_bits_embedded": embedded,
            "n_tokens": len(tokens),
            "syncpool": self.syncpool,
            # Degeneration telemetry. A document can survive generation and
            # still have spent most of itself in a repetition loop (zero-bit,
            # single-candidate steps); that shows up here as a large
            # n_singleton_steps and a long bits-per-token shortfall, and it is
            # what makes a record blow past the length target.
            "n_singleton_steps": enc_stats.get("n_singleton_steps"),
            "last_bit_step": enc_stats.get("last_bit_step"),
            # The ids as emitted, kept for the token channel: re-tokenizing
            # `text` does not always give them back (greedy BPE re-merges a pair
            # the sampler emitted separately), and that difference is the whole
            # reason the clean text channel is lossy without SyncPool.
            "token_ids": list(tokens),
            "bpe_roundtrip_exact": (
                # verbose=False: these ids are only compared against the
                # emitted ones, never run through the model, so HuggingFace's
                # "will result in indexing errors" warning does not apply.
                self.lm.tokenizer(text, verbose=False)["input_ids"] == list(tokens)
            ),
        }
        return text

    def recover_message(self, stego_text: str, token_ids=None, stats=None, **kwargs) -> Any:
        """Recover the payload from a stego text, or from its emitted ids.

        ``token_ids`` selects the **token channel** — the scheme itself, with no
        transport in the way, which a clean decode hits bit-exactly. Without it
        the ids come from re-tokenizing the text (the **text channel**), which is
        the realistic transport and, absent SyncPool, lossy even with no attack.
        ``stats`` optionally collects the decoder's silent-fallback counters;
        in a clean decode they must all be zero.
        """
        if self._error_encoded_length is None or self._context is None:
            raise ValueError(
                "DiscopSystem needs `error_encoded_length` and `context` set "
                "(run hide_message first, or restore them via system_state)."
            )
        context_ids = self.lm.context_ids(self._context)
        decode = db.decode_discop_syncpool if self.syncpool else db.decode_discop
        kwargs = {"syncpool_key": self._syncpool_key} if self.syncpool else {}
        n = self._error_encoded_length
        bits = decode(
            self.lm,
            stego_text,
            context_ids,
            self._seed_int,
            temp=self.temp,
            top_p=self.top_p,
            token_ids=token_ids,
            stats=stats,
            # Only the first `n` bits survive the slice below, so there is no
            # reason to buy any more. One step is one LM forward, and an
            # attacked text otherwise runs ~3x past the point the payload is
            # complete — see decode_discop_syncpool.
            max_bits=n,
            **kwargs,
        )
        raw = [int(b) for b in bits[:n]]
        # Pad short recoveries with ERASURE, not 0: those copies were never
        # received, and under the interleaved layout a truncated tail drops
        # copies of *every* message bit, so zero-padding would bias the whole
        # message toward 0 rather than adding unbiased noise.
        if len(raw) < n:
            raw = raw + [ERASURE] * (n - len(raw))
        decoded = self.ecc.decode(raw, n)
        return self.encoder.decode(decoded)
