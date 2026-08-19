"""Vendored Meteor steganography backend (in-house baseline only).

Meteor is the cryptographically-secure token-level scheme of Kaptchuk, Jois,
Green & Rubin (CCS 2021). This file adapts the modern-``transformers`` port at
https://github.com/rohanssrao/meteor (itself derived from the authors' original
Colab demo, https://gist.github.com/tusharjois/ec8603b711ff61e09167d8fef37c9b86)
so it drives a local HuggingFace causal LM through the same encode/decode path
the paper describes.

This module is a vendored dependency for an in-house baseline comparison; it is
deliberately isolated from the paper's own systems and imported only by
``MeteorSystem``. The one substantive change over the upstream port is that
special-token ids (end-of-text / block list) are derived from the tokenizer
rather than hardcoded to Qwen's vocabulary, so GPT-2 (the paper-faithful Meteor
channel) works correctly.
"""

from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# KV-cache crop length; GPT-2's positional limit is 1024.
MAX_CONTEXT_LENGTH = 1022
BLOCK_VALUE = -1e20  # additive logit penalty to forbid a token


class DRBG(object):
    """HMAC-SHA512 deterministic bit generator (the Meteor mask stream).

    Encode and decode both instantiate this with the same (key, seed), so the
    XOR mask is identical on both sides regardless of the exact byte schedule.
    """

    def __init__(self, key, seed):
        self.key = key
        self.val = b"\x01" * 64
        self.reseed(seed)
        self.byte_index = 0
        self.bit_index = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b""):
        self.key = self.hmac(self.key, self.val + b"\x00" + data)
        self.val = self.hmac(self.key, self.val)
        if data:
            self.key = self.hmac(self.key, self.val + b"\x01" + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        xs = np.zeros(n, dtype=bool)
        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1
            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1
            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)
        self.reseed()
        return xs


# Default symmetric material. The MeteorSystem wrapper passes its own key/nonce;
# these keep the free functions callable standalone for debugging.
sample_key = b"0x01" * 64
sample_seed_prefix = b"sample"
sample_nonce_counter = b"\x00" * 16


class TokenizerWrapper:
    """Expose the GPT-2 ``.encoder``/``.decoder`` interface Meteor expects."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.encoder = tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)


def limit_past(past):
    """Crop the KV cache to MAX_CONTEXT_LENGTH (handles DynamicCache + tuples).

    GPT-2 derives position ids from the cache length, so an uncropped cache walks
    the positions past the 1024-entry embedding table and raises IndexError. The
    cache layout has moved twice across transformers versions; all three are
    handled here because missing the right one makes this a silent no-op that only
    fails once a generation actually gets long (>1024 tokens).
    """
    if past is None:
        return None
    # transformers >= ~4.54: DynamicCache holds per-layer objects with .keys/.values
    if hasattr(past, "layers"):
        if past.get_seq_length() > MAX_CONTEXT_LENGTH:
            for layer in past.layers:
                layer.keys = layer.keys[:, :, -MAX_CONTEXT_LENGTH:, :]
                layer.values = layer.values[:, :, -MAX_CONTEXT_LENGTH:, :]
        return past
    # older DynamicCache: parallel key_cache/value_cache lists
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


def kl(q, logq, logp):
    res = q * (logq - logp) / 0.69315
    res[q == 0] = 0
    return res.sum().item()


def entropy(q, logq):
    res = q * logq / 0.69315
    res[q == 0] = 0
    return -res.sum().item()


def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ("{0:0%db}" % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return "." in token or "!" in token or "?" in token


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


class MeteorLM:
    """Loaded local causal LM plus its tokenizer wrapper and special-token ids."""

    def __init__(self, model_name: str = "gpt2", seed: int = 1234, device=None):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.enc = TokenizerWrapper(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model_name = model_name

        # Derive special tokens from the tokenizer (GPT-2: eos == 50256). Fall
        # back to the last vocab index if the tokenizer declares none.
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = len(self.enc.encoder) - 1
        self.endoftext_id = eos
        # Tokens Meteor forbids: end-of-text and (GPT-2) the double-newline id.
        block = {eos}
        newline2 = self.enc.encoder.get("ĊĊ")  # GPT-2 byte-level "\n\n"
        if newline2 is not None:
            block.add(newline2)
        self.block_ids = sorted(block)

    def encode_context(self, raw_text: str):
        return [self.endoftext_id] + self.enc.encode(raw_text)


def encode_meteor(
    lm: MeteorLM,
    message,
    context,
    finish_sent=False,
    temp=1.0,
    precision=16,
    topk=50000,
    input_key=sample_key,
    input_nonce=sample_nonce_counter,
):
    """Grow ``message`` (list of bits) into cover tokens. Returns token id list."""
    model, enc, device = lm.model, lm.enc, lm.device
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)
    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val]

    prev = context
    output_tokens = []
    past = None

    with torch.inference_mode():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            for bid in lm.block_ids:
                logits[0, -1, bid] = BLOCK_VALUE
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]
                indices = indices[:k]

                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[: overfill_index[0]]
                cum_probs += cur_int_range - cum_probs[-1]
                cum_probs += cur_interval[0]

                message_bits = message[i : i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))

                mask_bits = mask_generator.generate_bits(precision)
                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))

                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

            prev = indices[selection].view(1)
            output_tokens.append(prev.item())

            partial = enc.decode(output_tokens)
            if "<eos>" in partial:
                break

    return output_tokens


def decode_meteor(
    lm: MeteorLM,
    text,
    context,
    temp=1.0,
    precision=16,
    topk=50000,
    input_key=sample_key,
    input_nonce=sample_nonce_counter,
):
    """Recover the bit stream Meteor embedded in ``text``."""
    model, enc, device = lm.model, lm.enc, lm.device
    inp = enc.encode(text)

    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)

    max_val = 2**precision
    cur_interval = [0, max_val]

    prev = context
    past = None
    message = []
    with torch.inference_mode():
        i = 0
        while i < len(inp):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            for bid in lm.block_ids:
                logits[0, -1, bid] = BLOCK_VALUE
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]

            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[: overfill_index[0]]
                k = overfill_index[0].item()
            cum_probs += cur_int_range - cum_probs[-1]
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # BPE mismatch heuristic (upstream): re-align to a prefix/longer token.
            if rank >= k:
                true_token_text = enc.decoder.get(inp[i], "")
                for rank_idx in range(k):
                    prop_token_text = enc.decoder.get(indices[rank_idx].item(), "")
                    if len(prop_token_text) <= len(true_token_text) and prop_token_text == true_token_text[: len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1 : i + 1] = enc.encode(suffix)
                        break
                    elif len(prop_token_text) > len(true_token_text) and true_token_text == prop_token_text[: len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder.get(inp[i + num_extra], "")
                            num_extra += 1
                        if prop_token_text == whole_text[: len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]
                            if len(whole_text) > len(prop_token_text):
                                inp[i + 1 : i + 1] = enc.encode(whole_text[len(prop_token_text):])
                            break
                else:
                    rank = 0

            selection = rank

            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))

            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return message
