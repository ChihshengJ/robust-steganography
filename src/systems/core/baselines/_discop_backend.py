"""Vendored Discop steganography backend (in-house baseline only).

Discop is the "distribution copies" provably-secure scheme of Ding, Chen, Wang,
Zhao, Zhang & Yu (IEEE S&P 2023). The official implementation
(https://github.com/comydream/Discop) has its hot path in Cython/C++
(``stega_cy.pyx``, a Huffman tree over ``shared_ptr<Node>``). This module is a
faithful pure-Python port of the text/GPT-2 path only — no Cython build step, no
image/TTS variants.

Determinism, as in the original: encode and decode both seed Python's ``random``
with the same key and walk the *same* Huffman tree along the path to the sampled
token, so the ``random.random()`` draws stay in lockstep and the recovered bits
match. The seed therefore plays the role of the shared symmetric key.
"""

from __future__ import annotations

import random
from math import log2

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_CONTEXT_LENGTH = 1022

# GPT-2 token ids Discop forbids so that decoding stays unambiguous (from the
# upstream utils.filter_out_indices_gpt): endoftext, single/double newline, and
# the leading-space punctuation that collides with its bare form under BPE.
GPT2_FILTER_IDS = (50256, 198, 628, 764, 837)

# The SyncPool variant of the same list. Every entry above except endoftext is
# there for *segmentation disambiguation* — which is precisely what SyncPool
# provides, by grouping prefix-related candidates into one ambiguity pool and
# having the receiver walk bytes instead of re-tokenizing. Under SyncPool the
# newline bans are therefore not buying anything, and they cost a great deal:
# they force GPT-2 to emit a length-matched document (~750 tokens) as one
# unbroken run-on paragraph, which is far off-distribution and drives it into a
# repetition loop. That is fatal rather than merely ugly, because `top_p` then
# truncates the nucleus to a *single* candidate (see `_probs_indices`), leaving
# one ambiguity pool, a one-leaf Huffman tree, and hence zero embedded bits and
# zero freedom to escape — an absorbing, zero-capacity state that burns the
# whole generation budget. Measured on the trajectory that first exposed this
# (capacity 16, r=117): 779/1872 bits in 2200 tokens and permanently stuck with
# the ban, 1872/1872 bits in 988 tokens without it.
#
# ` .` and ` ,` are kept only because the newline lift is what was measured;
# they are redundant under SyncPool by the same argument and can go too.
#
# This does not touch Discop's security argument: the sampler still draws from
# the model's own conditional. It removes an ad-hoc edit *to* that conditional.
GPT2_FILTER_IDS_SYNCPOOL = (50256, 764, 837)


class Node:
    __slots__ = ("prob", "left", "right", "index", "search_path")

    def __init__(self, prob, left, right, index, search_path):
        self.prob = prob
        self.left = left
        self.right = right
        self.index = index  # >=0 leaf token id, -1 internal
        self.search_path = search_path  # 0 here, -1 left, 1 right, 9 unknown


def _is_leaf(node: Node) -> bool:
    return node.index != -1


def create_huffman_tree(indices, probs, search_for):
    """Two-queue O(n) Huffman build over descending-sorted (indices, probs)."""
    from collections import deque

    q1 = deque()
    q2 = deque()
    for i in range(len(indices) - 1, -1, -1):
        search_path = 0 if search_for == indices[i] else 9
        q1.append(Node(probs[i], None, None, indices[i], search_path))

    def _pop_smaller():
        if q1 and q2:
            src = q1 if q1[0].prob < q2[0].prob else q2
        elif q1:
            src = q1
        else:
            src = q2
        return src.popleft()

    while len(q1) + len(q2) > 1:
        first = _pop_smaller()
        second = _pop_smaller()
        prob = first.prob + second.prob
        search_path = 9
        if first.search_path != 9:
            search_path = -1
        elif second.search_path != 9:
            search_path = 1
        q2.append(Node(prob, first, second, -1, search_path))

    return q2[0] if q2 else q1[0]


def encode_step(indices, probs, message_bits, bit_cursor):
    """Discop encode for one token. Returns (sampled_index, n_bits_embedded).

    ``message_bits`` is the full bit list; ``bit_cursor`` is how many bits have
    already been embedded. Bits beyond the message are treated as 0 (padding).
    """
    node = create_huffman_tree(indices, probs, -1)
    n_bits = 0
    n_msg = len(message_bits)
    while not _is_leaf(node):
        prob_sum = node.prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum
        partition = node.left.prob
        p0 = -1 if ptr_0 < partition else 1
        p1 = -1 if ptr_1 < partition else 1

        pos = bit_cursor + n_bits
        bit = message_bits[pos] if pos < n_msg else 0  # pad exhausted message with 0
        chosen = p1 if bit == 1 else p0
        node = node.right if chosen == 1 else node.left

        if p0 != p1:
            n_bits += 1
    return node.index, n_bits


def decode_step(indices, probs, stego_t):
    """Discop decode for one token. Returns list of recovered bits (ints).

    Raises ValueError on an undecodable step (upstream sentinel ``'x'``).
    """
    node = create_huffman_tree(indices, probs, stego_t)
    bits = []
    while not _is_leaf(node):
        prob_sum = node.prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum
        partition = node.left.prob
        p0 = -1 if ptr_0 < partition else 1
        p1 = -1 if ptr_1 < partition else 1

        if p0 != p1:  # this node embeds a bit
            if node.search_path == 9:
                raise ValueError("Discop: failed to decode step")
            if p0 == -1:
                swap = {-1: 0, 1: 1}
            else:
                swap = {-1: 1, 1: 0}
            bits.append(swap[node.search_path])
            node = node.left if node.search_path == -1 else node.right
        else:
            node = node.left if p0 == -1 else node.right

    if node.search_path != 0:
        raise ValueError("Discop: could not reach target leaf")
    return bits


def _limit_past(past):
    """Crop the KV cache so GPT-2's cache-derived position ids stay under 1024.

    The cache layout differs across transformers versions, and handling only one
    of them makes this a silent no-op that fails with an IndexError once a
    generation exceeds the positional limit — hence the three branches.
    """
    if past is None:
        return None
    if hasattr(past, "layers"):
        if past.get_seq_length() > MAX_CONTEXT_LENGTH:
            for layer in past.layers:
                layer.keys = layer.keys[:, :, -MAX_CONTEXT_LENGTH:, :]
                layer.values = layer.values[:, :, -MAX_CONTEXT_LENGTH:, :]
        return past
    if hasattr(past, "get_seq_length"):
        if past.get_seq_length() > MAX_CONTEXT_LENGTH and hasattr(past, "key_cache"):
            for i in range(len(past.key_cache)):
                past.key_cache[i] = past.key_cache[i][:, :, -MAX_CONTEXT_LENGTH:, :]
                past.value_cache[i] = past.value_cache[i][:, :, -MAX_CONTEXT_LENGTH:, :]
        return past
    new_past = []
    for key, value in past:
        new_past.append(
            (key[:, :, -MAX_CONTEXT_LENGTH:, :], value[:, :, -MAX_CONTEXT_LENGTH:, :])
        )
    return tuple(new_past)


class DiscopLM:
    """Loaded GPT-2-style causal LM + tokenizer for the Discop text channel."""

    def __init__(self, model_name: str = "gpt2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model_name = model_name

    def context_ids(self, text: str):
        ids = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )["input_ids"].to(self.device)
        return ids

    @torch.no_grad()
    def _probs_indices(self, prev, past, temp, top_p, filter_ids=GPT2_FILTER_IDS):
        """One step's (probs, indices, past). Encoder and decoder both call this.

        ``filter_ids`` must match on both sides or the candidate sets differ and
        every recovered bit is chance; the SyncPool paths pass
        ``GPT2_FILTER_IDS_SYNCPOOL`` and the plain paths take the default.
        """
        past = _limit_past(past)
        out = self.model(prev, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        for fid in filter_ids:
            if fid < logits.shape[0]:
                logits[fid] = -1e10
        logits, indices = logits.sort(descending=True)
        logits = logits.double()
        probs = F.softmax(logits / temp, dim=-1)
        if top_p is not None and 0 < top_p < 1.0:
            cum = probs.cumsum(0)
            k = (cum > top_p).nonzero()[0].item() + 1
            probs = probs[:k]
            indices = indices[:k]
            probs = probs / cum[k - 1]
        return probs.tolist(), indices.tolist(), past


def encode_discop(lm: DiscopLM, message_bits, context_ids, seed, temp=1.0, top_p=0.92, max_length=512):
    """Grow message bits into GPT-2 tokens. Stops once all bits are embedded.

    Returns (generated_token_ids, n_bits_embedded). ``n_bits_embedded`` short of
    ``len(message_bits)`` means ``max_length`` cut the generation off before the
    payload fit — the trailing bits were never written and no attack is needed
    to lose them. Callers must check; :meth:`DiscopSystem.hide_message` does.

    ``max_length`` is a budget, not a model limit: ``_limit_past`` crops the KV
    cache, so generation runs past GPT-2's 1024 positional slots fine.
    """
    random.seed(seed)
    past = None
    prev = context_ids
    generated = []
    n_msg = len(message_bits)
    embedded = 0
    for _ in range(max_length):
        probs, indices, past = lm._probs_indices(prev, past, temp, top_p)
        idx, n_bits = encode_step(indices, probs, message_bits, embedded)
        embedded += n_bits
        generated.append(idx)
        prev = torch.tensor([idx], device=lm.device).unsqueeze(0)
        if embedded >= n_msg:
            break
    return generated, embedded


def decode_discop(lm: DiscopLM, stego_text, context_ids, seed, temp=1.0, top_p=0.92,
                  token_ids=None, stats=None, max_bits=None):
    """Recover the embedded bit stream from a stego text or its token ids.

    Two channels: pass ``token_ids``
    for the **token channel** (the scheme with no transport in the way, bit-exact
    when clean), or ``stego_text`` for the **text channel**, where the ids come
    back from re-tokenizing. The text channel is lossy — GPT-2's greedy BPE
    re-merges a pair the sampler emitted separately (``[" pay", "ments"]`` ->
    ``[" payments"]``), and from that step the decoder walks a different token
    sequence than the encoder did.

    Both fallbacks below are silent desyncs, so ``stats`` (a dict) collects
    ``out_of_support`` / ``undecodable`` / ``n_tokens``. Each substitutes exactly
    one bit where ``decode_step`` would have returned a variable number, and
    neither consumes the ``random.random()`` draws the encoder spent on that
    token — so the shared RNG stream is offset from there on too. That is
    acceptable only because it fires on attacked text, which is already at
    chance; in a clean decode these must be zero.

    ``max_bits`` stops the walk once that many bits are out. The caller
    discards everything past ``error_encoded_length`` anyway (see
    :meth:`DiscopSystem.recover_message`), so this changes no recovered bit —
    it only declines to pay for bits that are about to be thrown away. It
    matters because each step is one LM forward (~250 ms for gpt2-medium on
    CPU) and an attacked text runs far longer than the payload needs: the
    stegotext was sized so the *encoder* just fit the payload, while the
    decoder's byte walk re-reads an attacked text at a different bytes-per-step
    rate and would otherwise keep going to the last byte.
    """
    random.seed(seed)
    if token_ids is not None:
        stego = list(token_ids)
    else:
        # No max_length/truncation: cutting the ids at 1024 would silently
        # discard the tail of any longer stego text.
        # verbose=False silences HuggingFace's "longer than the specified
        # maximum sequence length ... will result in indexing errors" warning.
        # It does not apply: 1024 is GPT-2's positional limit, and this loop
        # feeds the model one token at a time with the KV cache cropped by
        # _limit_past(), so position ids stay in range at any document length.
        stego = lm.tokenizer(stego_text, return_tensors="pt", verbose=False)[
            "input_ids"
        ][0].tolist()
    if stats is not None:
        stats.setdefault("out_of_support", 0)
        stats.setdefault("undecodable", 0)
        stats["n_tokens"] = len(stego)

    past = None
    prev = context_ids
    bits = []
    for t in range(len(stego)):
        probs, indices, past = lm._probs_indices(prev, past, temp, top_p)
        if stego[t] not in indices:
            # Attacked/paraphrased text can produce a token outside the top-p
            # support; the step is undecodable. Emit a placeholder bit and move
            # on rather than aborting, so BER stays well-defined.
            bits.append(0)
            if stats is not None:
                stats["out_of_support"] += 1
        else:
            try:
                bits.extend(decode_step(indices, probs, stego[t]))
            except ValueError:
                bits.append(0)
                if stats is not None:
                    stats["undecodable"] += 1
        if max_bits is not None and len(bits) >= max_bits:
            break
        prev = torch.tensor([stego[t]], device=lm.device).unsqueeze(0)
    return bits


# ---------------------------------------------------------------------------
# SyncPool variants (Qi et al., arXiv:2403.17524)
# ---------------------------------------------------------------------------
#
# Discop is the scheme Qi et al. validated SyncPool on, so this is the faithful
# combination rather than an extension. The Huffman tree is built over
# *ambiguity pools* instead of tokens — each pool a single leaf carrying its
# members' summed probability — and which member is emitted is drawn from a
# keyed CSPRNG the receiver shares. Pooling leaves the marginal distribution over
# tokens unchanged, so Discop's provable-security argument is untouched; what it
# removes is the decoder's need to re-tokenize.
#
# Kept separate from the plain functions above so pre-SyncPool records still
# decode exactly as they did.


def encode_discop_syncpool(
    lm: DiscopLM,
    message_bits,
    context_ids,
    seed,
    temp=1.0,
    top_p=0.92,
    max_length=512,
    syncpool_key=b"\x00" * 32,
    stats=None,
):
    """Discop + SyncPool. Returns ``(generated_token_ids, n_bits_embedded)``.

    ``n_bits_embedded`` short of ``len(message_bits)`` means ``max_length`` cut
    generation off before the payload fit; :meth:`DiscopSystem.hide_message`
    raises on that rather than emitting a record whose tail was never written.
    SyncPool lowers bits/token (within-pool choices no longer carry payload), so
    the budget has to be larger than it was without it.

    ``stats`` (a dict) optionally collects why a generation ran long, which is
    the difference between the two ways this can fail. ``n_singleton_steps``
    counts steps whose nucleus held exactly one ambiguity pool: those embed zero
    bits *and* leave the sampler no choice, so a run of them is GPT-2 stuck in a
    repetition loop rather than a budget that is merely too small.
    ``last_bit_step`` is the step at which the last payload bit went in — far
    from the end means the tail was dead weight.
    """
    from ._syncpool import SyncPRNG, segmentation_index

    index = segmentation_index(lm.tokenizer)
    prng = SyncPRNG(syncpool_key)
    random.seed(seed)

    past = None
    prev = context_ids
    generated = []
    n_msg = len(message_bits)
    embedded = 0
    singleton = 0
    last_bit_step = -1
    for step in range(max_length):
        probs, indices, past = lm._probs_indices(prev, past, temp, top_p, GPT2_FILTER_IDS_SYNCPOOL)
        pools = index.pools(indices, probs)
        pool_ids = list(range(len(pools)))
        pool_probs = pools.probs.tolist()

        pool_idx, n_bits = encode_step(pool_ids, pool_probs, message_bits, embedded)
        embedded += n_bits
        if len(pools) == 1:
            singleton += 1
        if n_bits:
            last_bit_step = step

        token = pools.sync_sample(prng, pool_idx)
        generated.append(token)
        prev = torch.tensor([token], device=lm.device).unsqueeze(0)
        if embedded >= n_msg:
            break
    if stats is not None:
        stats["n_singleton_steps"] = singleton
        stats["last_bit_step"] = last_bit_step
        stats["n_steps"] = len(generated)
    return generated, embedded


def decode_discop_syncpool(
    lm: DiscopLM,
    stego_text,
    context_ids,
    seed,
    temp=1.0,
    top_p=0.92,
    token_ids=None,
    stats=None,
    syncpool_key=b"\x00" * 32,
    max_bits=None,
):
    """Recover Discop+SyncPool bits from a stegotext (or from the emitted ids).

    The text channel walks bytes rather than calling the tokenizer: the candidate
    that is a prefix of the remaining stegotext identifies an ambiguity pool
    (every candidate that could match lies in the same one), the pool yields the
    message bits, and the shared CSPRNG replays which member the sender emitted,
    which says how many bytes to consume. A clean stegotext therefore decodes
    bit-exactly — the property the plain text channel cannot have.

    On attacked text no candidate matches; the highest-mass pool stands in,
    ``stats['no_prefix_match']`` counts it, and the walk continues so the bit
    vector stays well defined. Every bit from the first such step is chance.

    ``max_bits`` stops the walk once that many bits are out — see
    :func:`decode_discop`. It is result-preserving (the caller slices to
    ``error_encoded_length``) and worth a great deal here: on an attacked text
    the byte walk advances by the *sync-sampled* member of the matched pool,
    which is not the token the attacker actually wrote, so it consumes ~2.3
    bytes per step against the clean walk's ~4.3 and would run roughly three
    times as many forwards as the payload needs. Measured on
    ``local_paraphrase`` records at capacity 16: 2064 bits complete by step
    ~510-560, against ~1465 steps to reach the last byte.

    Note that this redefines ``stats['n_steps']`` and ``stats['no_prefix_match']``
    as counts *up to payload completion* rather than over the whole text. Any
    analysis of those should normalise by ``n_steps`` rather than compare raw
    counts against records decoded before this existed.
    """
    from ._syncpool import SyncPRNG, segmentation_index

    index = segmentation_index(lm.tokenizer)
    prng = SyncPRNG(syncpool_key)
    random.seed(seed)

    if stats is not None:
        for key in ("n_steps", "no_prefix_match", "undecodable"):
            stats.setdefault(key, 0)

    on_token_channel = token_ids is not None
    ids = list(token_ids) if on_token_channel else []
    data = b"" if on_token_channel else stego_text.encode("utf-8")

    past = None
    prev = context_ids
    bits = []
    offset = 0
    step = 0

    while (step < len(ids)) if on_token_channel else (offset < len(data)):
        probs, indices, past = lm._probs_indices(prev, past, temp, top_p, GPT2_FILTER_IDS_SYNCPOOL)
        pools = index.pools(indices, probs)
        pool_ids = list(range(len(pools)))
        pool_probs = pools.probs.tolist()

        observed = None
        if on_token_channel:
            pool_idx = index.pool_of_token(pools, ids[step])
        else:
            pool_idx = index.pool_of_prefix(pools, data, offset)
            if pool_idx is None:
                # Attacked text. Take the bits from the highest-mass pool but
                # step over a real vocabulary token, so the walk stays on token
                # boundaries rather than crawling the text a byte at a time.
                observed = index.longest_vocab_prefix(data, offset)
        if pool_idx is None:
            pool_idx = 0
            if stats is not None:
                stats["no_prefix_match"] += 1

        try:
            bits.extend(decode_step(pool_ids, pool_probs, pool_idx))
        except ValueError:
            bits.append(0)
            if stats is not None:
                stats["undecodable"] += 1

        token = pools.sync_sample(prng, pool_idx)
        if on_token_channel:
            token = ids[step]
        else:
            # Advance by the sync-sampled token: that is the sender's choice,
            # and replaying it is what holds the two token sequences together.
            # `observed` is set only when nothing matched at all.
            if observed is not None:
                token = observed
            offset += max(1, min(len(index.surfaces[token]), len(data) - offset))
        step += 1

        prev = torch.tensor([token], device=lm.device).unsqueeze(0)
        if stats is not None:
            stats["n_steps"] += 1
        if max_bits is not None and len(bits) >= max_bits:
            break

    return bits
