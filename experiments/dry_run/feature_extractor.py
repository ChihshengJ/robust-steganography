"""Self-contained statistical feature extraction for stegoanalysis.

Computes 16 features per text: perplexity (4), token distribution (4),
n-gram (3), sentence-level (3), and lexical (2).
"""

from __future__ import annotations

import math
from collections import Counter

import nltk
import numpy as np
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure NLTK data is available
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_STOPWORDS: set[str] = set(nltk.corpus.stopwords.words("english"))


class FeatureExtractor:
    """Compute 16 statistical features for a single text."""

    def __init__(self, ppl_model: str = "gpt2-large", device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._tok = AutoTokenizer.from_pretrained(ppl_model)
        self._model = AutoModelForCausalLM.from_pretrained(ppl_model).to(device)
        self._model.eval()
        self._max_len = self._model.config.max_position_embeddings
        self._tiktoken = tiktoken.get_encoding("o200k_base")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, float]:
        """Return dict of 16 named features."""
        feats: dict[str, float] = {}
        feats.update(self._perplexity_features(text))
        feats.update(self._token_distribution_features(text))
        feats.update(self._ngram_features(text))
        feats.update(self._sentence_features(text))
        feats.update(self._lexical_features(text))
        return feats

    def extract_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.extract(t) for t in texts]

    # ------------------------------------------------------------------
    # Perplexity (4 features)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _perplexity_features(self, text: str) -> dict[str, float]:
        encodings = self._tok(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            return {
                "perplexity": float("inf"),
                "mean_nll": float("inf"),
                "std_nll": 0.0,
                "max_nll": float("inf"),
            }

        nlls: list[torch.Tensor] = []
        stride = self._max_len // 2
        for begin in range(0, seq_len, stride):
            end = min(begin + self._max_len, seq_len)
            chunk_ids = input_ids[:, begin:end]
            outputs = self._model(chunk_ids, labels=chunk_ids)

            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = chunk_ids[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_nlls = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            offset = max(0, (begin + stride - begin - 1)) if begin > 0 else 0
            nlls.append(token_nlls[offset:].cpu())

            if end >= seq_len:
                break

        all_nlls = torch.cat(nlls)
        mean_nll = all_nlls.mean().item()
        return {
            "perplexity": math.exp(mean_nll),
            "mean_nll": mean_nll,
            "std_nll": all_nlls.std().item(),
            "max_nll": all_nlls.max().item(),
        }

    # ------------------------------------------------------------------
    # Token distribution (4 features)
    # ------------------------------------------------------------------

    def _token_distribution_features(self, text: str) -> dict[str, float]:
        token_ids = self._tiktoken.encode(text)
        n = len(token_ids)
        if n == 0:
            return {
                "token_count": 0,
                "type_token_ratio": 0.0,
                "mean_token_length": 0.0,
                "token_freq_entropy": 0.0,
            }

        unique = len(set(token_ids))
        # Decode each token to get char lengths
        lengths = [len(self._tiktoken.decode([tid])) for tid in token_ids]

        # Shannon entropy of token frequency distribution
        counts = Counter(token_ids)
        probs = np.array(list(counts.values()), dtype=float) / n
        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

        return {
            "token_count": float(n),
            "type_token_ratio": unique / n,
            "mean_token_length": float(np.mean(lengths)),
            "token_freq_entropy": entropy,
        }

    # ------------------------------------------------------------------
    # N-gram (3 features)
    # ------------------------------------------------------------------

    @staticmethod
    def _ngram_features(text: str) -> dict[str, float]:
        words = text.lower().split()
        if len(words) < 3:
            return {
                "bigram_repetition_rate": 0.0,
                "trigram_repetition_rate": 0.0,
                "bigram_entropy": 0.0,
            }

        bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]

        bi_counts = Counter(bigrams)
        tri_counts = Counter(trigrams)

        bi_repeated = sum(1 for c in bi_counts.values() if c > 1)
        tri_repeated = sum(1 for c in tri_counts.values() if c > 1)

        # Bigram entropy
        n_bi = len(bigrams)
        bi_probs = np.array(list(bi_counts.values()), dtype=float) / n_bi
        bi_entropy = -float(np.sum(bi_probs * np.log2(bi_probs + 1e-12)))

        return {
            "bigram_repetition_rate": bi_repeated / len(bi_counts) if bi_counts else 0.0,
            "trigram_repetition_rate": tri_repeated / len(tri_counts) if tri_counts else 0.0,
            "bigram_entropy": bi_entropy,
        }

    # ------------------------------------------------------------------
    # Sentence-level (3 features)
    # ------------------------------------------------------------------

    @staticmethod
    def _sentence_features(text: str) -> dict[str, float]:
        sents = nltk.sent_tokenize(text)
        if not sents:
            return {
                "sentence_count": 0.0,
                "mean_sentence_length": 0.0,
                "std_sentence_length": 0.0,
            }

        lengths = [len(s.split()) for s in sents]
        return {
            "sentence_count": float(len(sents)),
            "mean_sentence_length": float(np.mean(lengths)),
            "std_sentence_length": float(np.std(lengths)),
        }

    # ------------------------------------------------------------------
    # Lexical (2 features)
    # ------------------------------------------------------------------

    @staticmethod
    def _lexical_features(text: str) -> dict[str, float]:
        words = text.lower().split()
        if not words:
            return {
                "stopword_proportion": 0.0,
                "hapax_legomena_ratio": 0.0,
            }

        n = len(words)
        stopword_count = sum(1 for w in words if w in _STOPWORDS)
        counts = Counter(words)
        hapax = sum(1 for c in counts.values() if c == 1)

        return {
            "stopword_proportion": stopword_count / n,
            "hapax_legomena_ratio": hapax / len(counts) if counts else 0.0,
        }
