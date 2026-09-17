"""SyncPool — segmentation-ambiguity elimination for the token-level baseline.

Implements the scheme of Qi, Chen, Zeng, Zhang & Yu, *Provably Secure
Disambiguating Neural Linguistic Steganography* (arXiv:2403.17524), which fixes
the one failure token-level schemes like Discop have with no attacker present:
the sender must detokenize before transmitting, and ``token ids -> text -> token
ids`` is not the identity under greedy BPE. Emit ``[" bal", "cony"]``, write out
" balcony", and the receiver re-tokenizes to ``[" balcony"]`` — from that step it
walks a different token sequence than the encoder did and every later bit is
chance.

The fix, in three parts:

1. **Group.** Before the steganographic sampler runs, partition the step's
   candidate pool into *ambiguity pools*: maximal groups of candidates related by
   the prefix relation on their byte surfaces. ``" bal"`` and ``" balcony"`` land
   in one pool.
2. **Embed over pools.** The sampler (for Discop, the Huffman tree) sees each
   pool as a single candidate carrying the summed probability of its members. No
   message bit is spent distinguishing members of a pool — which is exactly the
   distinction the receiver cannot make.
3. **Sync.** Which member is actually emitted is drawn from a keyed CSPRNG that
   sender and receiver share, so the receiver reproduces the choice rather than
   inferring it. The marginal distribution over tokens is unchanged, so the
   security argument of the underlying scheme carries over untouched.

The receiver never calls the tokenizer. It walks the stegotext *bytes*, finds the
candidate that is a prefix of what remains (all such candidates lie in one pool —
see :meth:`SegmentationIndex.pool_of_prefix`), reads the message bits off the
pool, replays the CSPRNG draw to learn which member was sent, and consumes that
many bytes.

Cost: bits/token drops, because within-pool choices no longer carry payload. For
the length-matched baseline runs that changes the repetition rate, which
``calibrate_repetitions`` measures rather than assumes.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SyncPRNG",
    "SegmentationIndex",
    "AmbiguityPools",
    "segmentation_index",
    "derive_syncpool_key",
]


def derive_syncpool_key(passphrase: str) -> bytes:
    """Separate the SyncPool stream from the scheme's own key material.

    Discop's tree-walk RNG is keyed off the shared passphrase; drawing the
    within-pool choices from an independent stream keeps this change from
    perturbing the scheme's own randomness, so a SyncPool run and a plain run at
    the same key differ only where they are meant to.
    """
    return hashlib.sha256(b"syncpool|v1|" + passphrase.encode()).digest()


class SyncPRNG:
    """HMAC-SHA256 in counter mode — the shared CSPRNG of the paper's SyncSample.

    Sender and receiver instantiate this with the same key and consume draws at
    the same points (one per *ambiguous* pool selection; a singleton pool costs
    nothing), so the streams stay aligned without any side channel.
    """

    __slots__ = ("_key", "_counter")

    def __init__(self, key: bytes) -> None:
        self._key = key
        self._counter = 0

    def random(self) -> float:
        block = hmac.new(
            self._key, self._counter.to_bytes(8, "big"), hashlib.sha256
        ).digest()
        self._counter += 1
        # 56 bits is well past float64's 53-bit mantissa; the truncation is the
        # same on both sides, which is all that matters here.
        return int.from_bytes(block[:7], "big") / float(1 << 56)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte <-> printable-codepoint table (from the original BPE code)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@dataclass(frozen=True)
class AmbiguityPools:
    """One step's candidate pool, grouped and sorted by descending pool mass.

    ``probs[g]`` is the total probability of pool *g*. Its members are
    ``ids[starts[g]:ends[g]]`` with probabilities ``member_probs[starts[g]:...]``,
    held in lexicographic order so that both sides enumerate them identically.
    """

    probs: np.ndarray  # (n_pools,) float64, descending
    ids: np.ndarray  # (n_candidates,) int64, lexicographic by byte surface
    member_probs: np.ndarray  # (n_candidates,) float64, aligned with `ids`
    starts: np.ndarray  # (n_pools,) int64, index into `ids`
    ends: np.ndarray  # (n_pools,) int64
    _lex_pos: np.ndarray  # (n_candidates,) int64, global lexicographic ranks
    _pool_of: np.ndarray  # (n_candidates,) int64, candidate -> pool index

    def __len__(self) -> int:
        return len(self.probs)

    def members(self, pool: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = int(self.starts[pool]), int(self.ends[pool])
        return self.ids[s:e], self.member_probs[s:e]

    def sync_sample(self, prng: SyncPRNG, pool: int) -> int:
        """Draw the token actually emitted from pool *g* (the paper's SyncSample).

        A singleton pool is determined without consuming a draw, exactly as in
        the paper — and, more to the point, identically on both sides, which is
        what keeps the two PRNG streams aligned.
        """
        ids, probs = self.members(pool)
        if len(ids) == 1:
            return int(ids[0])
        total = float(probs.sum())
        r = prng.random() * total
        acc = 0.0
        for tid, p in zip(ids.tolist(), probs.tolist()):
            acc += p
            if r < acc:
                return int(tid)
        return int(ids[-1])


class SegmentationIndex:
    """Vocabulary-wide structure the pooling and the byte walk are built on.

    Everything here depends only on the tokenizer, so it is computed once per
    model and reused for every step of every document.

    The load-bearing piece is ``reach``. Sort the whole vocabulary by byte
    surface; then the tokens having token *x* as a prefix occupy a contiguous
    range starting at *x*, and ``reach[i]`` is where that range ends. Those
    ranges are *laminar* — two of them are nested or disjoint, because two
    tokens that share a prefix-descendant are themselves prefix-related — which
    is what makes the grouping in :meth:`pools` a vectorised running maximum
    instead of a per-step loop over tens of thousands of candidates.
    """

    def __init__(self, tokenizer) -> None:
        byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}
        vocab = tokenizer.get_vocab()
        size = max(vocab.values()) + 1
        surfaces: list[bytes] = [b""] * size
        for token, tid in vocab.items():
            try:
                surfaces[tid] = bytes(byte_decoder[ch] for ch in token)
            except KeyError:
                # Added/special tokens (``<|endoftext|>``) are not byte-level.
                # They never appear in a stegotext — the backend blocks them —
                # so any injective placeholder does.
                surfaces[tid] = token.encode("utf-8")
        self.surfaces = surfaces
        self.size = size

        order = sorted(range(size), key=surfaces.__getitem__)
        self.lex_ids = np.asarray(order, dtype=np.int64)
        lex_rank = np.empty(size, dtype=np.int64)
        lex_rank[self.lex_ids] = np.arange(size, dtype=np.int64)
        self.lex_rank = lex_rank

        # reach[i] = end of the lexicographic range of tokens prefixed by the
        # token at position i. Stack scan: entries above a stack entry are its
        # prefix-extensions, so a token that does not extend the top cannot
        # extend anything below it either.
        reach = np.empty(size, dtype=np.int64)
        stack: list[int] = []
        for i, tid in enumerate(order):
            surf = surfaces[tid]
            while stack and not surf.startswith(surfaces[order[stack[-1]]]):
                reach[stack.pop()] = i
            stack.append(i)
        while stack:
            reach[stack.pop()] = size
        self.reach = reach

        # Byte surface -> id, for the receiver's prefix lookup. Surfaces are
        # distinct in a byte-level vocabulary; keep the first on the off chance
        # an added token collides.
        surface_to_id: dict[bytes, int] = {}
        for tid, surf in enumerate(surfaces):
            if surf and surf not in surface_to_id:
                surface_to_id[surf] = tid
        self.surface_to_id = surface_to_id
        self.max_surface_len = max((len(s) for s in surfaces), default=1)

    # -- grouping ---------------------------------------------------------

    def pools(self, cand_ids: np.ndarray, cand_probs: np.ndarray) -> AmbiguityPools:
        """Partition one step's candidates into ambiguity pools.

        ``cand_ids`` are the step's candidates after the scheme's own truncation
        (for Discop, top-*p*) and ``cand_probs`` their
        probabilities; neither is required to be sorted. Pools come back ordered
        by descending mass, which is what both samplers expect of their input.
        """
        cand_ids = np.asarray(cand_ids, dtype=np.int64)
        cand_probs = np.asarray(cand_probs, dtype=np.float64)
        if cand_ids.size == 0:
            raise ValueError("SyncPool: empty candidate pool")

        pos = self.lex_rank[cand_ids]
        o = np.argsort(pos, kind="stable")
        pos = pos[o]
        ids = cand_ids[o]
        probs = cand_probs[o]

        n = pos.size
        # For candidate i, R[i] is the first candidate outside i's prefix range.
        # Laminarity makes "i starts a new pool" equivalent to "no earlier
        # candidate's range still covers i", i.e. a running maximum of R.
        r = np.searchsorted(pos, self.reach[pos], side="left")
        pivot = np.empty(n, dtype=bool)
        pivot[0] = True
        if n > 1:
            pivot[1:] = np.maximum.accumulate(r)[:-1] <= np.arange(1, n)
        group = np.cumsum(pivot) - 1
        starts = np.flatnonzero(pivot)
        ends = np.append(starts[1:], n)
        agg = np.bincount(group, weights=probs)

        order = np.argsort(-agg, kind="stable")
        inverse = np.empty(order.size, dtype=np.int64)
        inverse[order] = np.arange(order.size, dtype=np.int64)
        return AmbiguityPools(
            probs=agg[order],
            ids=ids,
            member_probs=probs,
            starts=starts[order],
            ends=ends[order],
            _lex_pos=pos,
            _pool_of=inverse[group],
        )

    # -- receiver-side byte walk ------------------------------------------

    def pool_of_prefix(self, pools: AmbiguityPools, data: bytes, offset: int) -> int | None:
        """Which pool holds the candidate that matches the text at ``offset``?

        Returns ``None`` when no candidate is a prefix of the remaining bytes —
        which on a clean stegotext cannot happen (the sender emitted one) and on
        an attacked one happens constantly, since the attacker rewrote the text.

        Any matching candidate identifies the pool, so the answer is well
        defined: if two candidates are both prefixes of the same string then one
        is a prefix of the other, and prefix-related candidates are pooled
        together by construction. The longest match is returned for determinism.
        """
        limit = min(self.max_surface_len, len(data) - offset)
        for length in range(limit, 0, -1):
            tid = self.surface_to_id.get(data[offset : offset + length])
            if tid is None:
                continue
            rank = self.lex_rank[tid]
            i = int(np.searchsorted(pools._lex_pos, rank, side="left"))
            if i < pools._lex_pos.size and pools._lex_pos[i] == rank:
                return int(pools._pool_of[i])
        return None

    def longest_vocab_prefix(self, data: bytes, offset: int) -> int | None:
        """Longest *vocabulary* token (candidate or not) matching at ``offset``.

        Only used when no candidate matches, i.e. on attacked text. Consuming a
        real token there keeps the byte walk on the tokenizer's own boundaries,
        so an attacked decode takes about as many steps as the text has tokens
        rather than roughly twice that — and its bit count stays comparable with
        the non-SyncPool decoder's, which tokenizes the attacked text outright.
        """
        limit = min(self.max_surface_len, len(data) - offset)
        for length in range(limit, 0, -1):
            tid = self.surface_to_id.get(data[offset : offset + length])
            if tid is not None:
                return tid
        return None

    def pool_of_token(self, pools: AmbiguityPools, token_id: int) -> int | None:
        """Which pool holds ``token_id``? ``None`` if it is not a candidate.

        Used by the token-channel decode, which is handed the ids the encoder
        emitted instead of having to find them in the text.
        """
        rank = self.lex_rank[token_id]
        i = int(np.searchsorted(pools._lex_pos, rank, side="left"))
        if i < pools._lex_pos.size and pools._lex_pos[i] == rank:
            return int(pools._pool_of[i])
        return None


_INDEX_CACHE: dict[tuple, SegmentationIndex] = {}


def segmentation_index(tokenizer) -> SegmentationIndex:
    """Cached :class:`SegmentationIndex` for a tokenizer (built once per model)."""
    key = (getattr(tokenizer, "name_or_path", type(tokenizer).__name__), len(tokenizer))
    index = _INDEX_CACHE.get(key)
    if index is None:
        index = SegmentationIndex(tokenizer)
        _INDEX_CACHE[key] = index
    return index
