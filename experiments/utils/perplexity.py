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
    def score(self, text: str) -> dict:
        """Compute perplexity of text.

        Returns {"perplexity": float, "mean_nll": float, "num_tokens": int}.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            return {"perplexity": float("inf"), "mean_nll": float("inf"), "num_tokens": seq_len}

        nlls = []
        stride = self.max_length // 2
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            target_begin = max(begin, 1) if begin == 0 else begin + stride - stride
            chunk_ids = input_ids[:, begin:end]

            outputs = self.model(chunk_ids, labels=chunk_ids)
            # Compute per-token NLL for the non-overlapping portion
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = chunk_ids[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_nlls = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Only keep tokens in the non-overlapping window
            offset = max(0, target_begin - begin - 1) if begin > 0 else 0
            nlls.append(token_nlls[offset:].cpu())

            if end >= seq_len:
                break

        all_nlls = torch.cat(nlls)
        mean_nll = all_nlls.mean().item()
        ppl = math.exp(mean_nll)
        return {"perplexity": ppl, "mean_nll": mean_nll, "num_tokens": seq_len}

    def score_batch(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """Score multiple texts. Processes sequentially (variable lengths)."""
        return [self.score(t) for t in texts]
