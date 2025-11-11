import re
from random import shuffle
from typing import List

from ..models.base import LanguageModel
from .attack import Attack


class NGramShuffleAttack(Attack):
    """Attack that breaks text into token n-grams and shuffles their positions."""

    def __init__(
        self,
        model: LanguageModel,
        n: int = 3,
    ):
        """
        Initialize the n-gram shuffle attack.

        Args:
            model: Language model whose tokenizer will be used
            n (int): Size of n-grams to use (default: 3)
            probability (float): Probability of shuffling each n-gram (0.0 to 1.0).
                               1.0 means shuffle all n-grams (default)
                               0.0 means no shuffling
                               0.5 means 50% of n-grams will be shuffled
            local (bool): If True, preserves sentence structure.
                         If False, shuffles token n-grams globally (default: True)
        """
        super().__init__()
        if n < 1:
            raise ValueError("n must be at least 1")

        self.model = model
        self.n = n

    def _split_into_ngrams(self, tokens: List[int]) -> List[List[int]]:
        """Split a list of tokens into n-grams."""
        ngrams = []
        for i in range(0, len(tokens), self.n):
            ngram = tokens[i : i + self.n]
            ngrams.append(ngram)
        return ngrams

    def _shuffle_ngrams(self, ngrams: List[List[int]], tp: float) -> List[List[int]]:
        """Shuffle a portion of n-grams based on probability."""
        num_to_shuffle = int(len(ngrams) * tp)
        if num_to_shuffle == 0:
            return ngrams

        # Get random indices to shuffle
        indices = list(range(len(ngrams)))
        shuffle(indices)
        indices = indices[:num_to_shuffle]

        # Create new list with shuffled n-grams
        shuffled = ngrams.copy()
        for idx1, idx2 in zip(sorted(indices), indices):
            shuffled[idx1] = ngrams[idx2]
        return shuffled

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the n-gram shuffle attack."""
        if not 0 <= tampering <= 1:
            raise ValueError("tp must be between 0 and 1")

        if tampering == 0:
            return text

        if local and tampering < 0.99:
            return self._local_shuffle(text, tampering)
        else:
            return self._global_shuffle(text, tampering)

    def _global_shuffle(self, text: str, tp: float) -> str:
        """Shuffle token n-grams globally."""
        # Convert text to tokens and split into n-grams
        tokens = self.model.tokenizer.encode(text)
        ngrams = self._split_into_ngrams(tokens)

        # Shuffle and decode
        shuffled_ngrams = self._shuffle_ngrams(ngrams, tp)
        shuffled_tokens = [token for ngram in shuffled_ngrams for token in ngram]
        result = self.model.tokenizer.decode(shuffled_tokens)
        print("Debug global shuffle:")
        print(f"in:\n{text}\nout:\n{result}")
        return result

    def _local_shuffle(self, text: str, tp: float) -> str:
        """Shuffle n-grams while preserving sentence structure."""
        # Split text into sentences while preserving separators
        parts = re.split(r"([.!?]+(?:\s+|$))", text)
        new_parts = []

        # parts[::2] are sentences, parts[1::2] are separators
        for i in range(0, len(parts), 2):
            sentence = parts[i]

            # Skip empty sentences
            if not sentence.strip():
                new_parts.append(sentence)
            else:
                # Convert sentence to tokens and split into n-grams
                tokens = self.model.tokenizer.encode(sentence)
                ngrams = self._split_into_ngrams(tokens)

                # Shuffle and decode
                shuffled_ngrams = self._shuffle_ngrams(ngrams, tp)
                shuffled_tokens = [
                    token for ngram in shuffled_ngrams for token in ngram
                ]
                new_parts.append(self.model.tokenizer.decode(shuffled_tokens))

            # Add the separator if it exists
            if i + 1 < len(parts):
                new_parts.append(parts[i + 1])
        result = "".join(new_parts)
        print("Debug local shuffle:")
        print(f"parts:\n{parts}\nnew_parts:\n{result}")
        return result
