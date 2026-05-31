import re
from random import choice, random

from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import (
    WordSwapEmbedding,
    WordSwapHowNet,
    WordSwapMaskedLM,
    WordSwapWordNet,
)

from .attack import Attack, iter_sentences_with_gaps


class SynonymAttack(Attack):
    """Attack that replaces words with synonyms while preserving formatting."""

    def __init__(
        self,
        method="wordnet",
        local_mode: bool | None = None,
    ):
        """
        Initialize the synonym attack with a specified method and swap probability.

        Arguments:
            method (str): The synonym replacement method to use. Options are:
                - "wordnet" (default): Uses WordNet for synonyms
                - "embedding": Uses word embeddings for similar words
                - "maskedlm": Uses masked language model for replacements
                - "hownet": Uses HowNet for synonyms
            local_mode: Controls local vs global attack behavior.
                - None (default): Use legacy behavior where tampering >= 0.99 forces global mode.
                - True: Force local mode (sentence-level) even at 100% tampering.
                - False: Force global mode regardless of tampering level.
                Note: For synonym attack, local mode processes each sentence independently,
                while global mode processes all words at once.
        """
        super().__init__()
        self.method = method

        # Select transformation based on the method
        if method == "wordnet":
            transformation = WordSwapWordNet()
        elif method == "embedding":
            transformation = WordSwapEmbedding()
        elif method == "maskedlm":
            transformation = WordSwapMaskedLM()
        elif method == "hownet":
            transformation = WordSwapHowNet()
        else:
            raise ValueError(f"Unsupported method: {method}")

        self.augmenter = Augmenter(transformation=transformation)

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the synonym replacement attack."""
        if not 0 <= tampering <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if tampering == 0:
            return text

        if self._resolve_local_mode(local, tampering):
            return self._local_synonym(text, tampering)
        else:
            return self._global_synonym(text, tampering)

    def _replace_words_in_text(self, text: str, tampering: float) -> str:
        """Replace words with synonyms based on tampering probability."""
        # Split text into words and whitespace, keeping both
        tokens = re.split(r"(\s+)", text)

        # Process only non-whitespace tokens
        new_tokens = []
        for token in tokens:
            if token.strip():  # If token is not whitespace
                # Randomly decide whether to try replacing this word
                if random() < tampering:
                    augmented_texts = self.augmenter.augment(token)
                    if augmented_texts:
                        single_word_synonyms = [
                            t for t in augmented_texts if len(t.split()) == 1
                        ]
                        if single_word_synonyms:
                            # Randomly select a synonym if available
                            new_tokens.append(choice(single_word_synonyms))
                            continue
                new_tokens.append(
                    token
                )  # Keep original if no replacement or probability check fails
            else:
                new_tokens.append(token)  # Keep whitespace as is

        return "".join(new_tokens)

    def _global_synonym(self, text: str, tampering: float) -> str:
        """Apply synonym replacement to the entire text at once."""
        result = self._replace_words_in_text(text, tampering)
        return result

    def _local_synonym(self, text: str, tampering: float) -> str:
        """Apply synonym replacement sentence by sentence."""
        new_parts: list[str] = []
        for gap, sentence in iter_sentences_with_gaps(text):
            new_parts.append(gap)
            if not sentence:
                continue
            if not sentence.strip():
                new_parts.append(sentence)
            else:
                new_parts.append(self._replace_words_in_text(sentence, tampering))
        return "".join(new_parts)
