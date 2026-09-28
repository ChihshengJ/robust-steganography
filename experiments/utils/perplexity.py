"""Perplexity computation using HuggingFace causal language models."""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerplexityScorer:
    """Compute token-level perplexity using a HuggingFace causal LM."""

    def __init__(self, model_name: str = "gpt2-large", device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings

    @torch.no_grad()
    def score(self, text: str, prefix: str | None = None) -> dict:
        """Perplexity of ``text``, conditioned on ``prefix`` when given.

        Every text token is scored exactly once. A text longer than the
        model's window is scored in windows that advance by half a window, each
        scoring only its new tokens with the preceding ones as context. The
        prefix (the task input x) is kept at the start of every window, so the
        whole text is conditioned on it, and is never scored itself. Without a
        prefix the text is conditioned on the BOS token, so its first token is
        scored too.

        Returns {"perplexity": float, "mean_nll": float, "num_tokens": int}.
        """
        text_ids = self.tokenizer(text).input_ids
        n = len(text_ids)
        if n == 0:
            return {"perplexity": float("inf"), "mean_nll": float("inf"), "num_tokens": 0}
        bos = [self.tokenizer.bos_token_id]
        prefix_ids = bos + (self.tokenizer(prefix + "\n\n").input_ids if prefix else [])
        # Keep at least half the window for the text.
        prefix_ids = prefix_ids[: self.max_length // 2]
        budget = self.max_length - len(prefix_ids)
        stride = max(1, budget // 2)

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        total_nll = 0.0
        scored_until = 0
        begin = 0
        while scored_until < n:
            end = min(begin + budget, n)
            window = torch.tensor(
                [prefix_ids + text_ids[begin:end]], device=self.device
            )
            logits = self.model(window).logits[0]
            # Text index k sits at window position p = len(prefix) + k - begin
            # and is predicted by the logits at p - 1.
            first = len(prefix_ids) + scored_until - begin
            last = len(prefix_ids) + end - begin
            total_nll += loss_fct(logits[first - 1 : last - 1], window[0, first:last]).item()
            scored_until = end
            begin += stride

        mean_nll = total_nll / n
        return {"perplexity": math.exp(mean_nll), "mean_nll": mean_nll, "num_tokens": n}

    def score_batch(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """Score multiple texts. Processes sequentially (variable lengths)."""
        return [self.score(t) for t in texts]
