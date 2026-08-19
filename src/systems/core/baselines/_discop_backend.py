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

    See _meteor_backend.limit_past — the cache layout differs across transformers
    versions, and handling only one makes this a no-op that fails at IndexError
    once a generation exceeds the positional limit.
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
    def _probs_indices(self, prev, past, temp, top_p):
        past = _limit_past(past)
        out = self.model(prev, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        for fid in GPT2_FILTER_IDS:
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

    Returns (generated_token_ids, n_bits_embedded).
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


def decode_discop(lm: DiscopLM, stego_text, context_ids, seed, temp=1.0, top_p=0.92):
    """Recover the embedded bit stream from stego text (re-tokenized to ids)."""
    random.seed(seed)
    stego = lm.tokenizer(
        stego_text, return_tensors="pt", max_length=1024, truncation=True
    )["input_ids"][0].tolist()

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
        else:
            try:
                bits.extend(decode_step(indices, probs, stego[t]))
            except ValueError:
                bits.append(0)
        prev = torch.tensor([stego[t]], device=lm.device).unsqueeze(0)
    return bits
