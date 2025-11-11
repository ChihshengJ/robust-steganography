import re
from random import choice, random

from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import (
    WordSwapEmbedding,
    WordSwapHowNet,
    WordSwapMaskedLM,
    WordSwapWordNet,
)

from .attack import Attack


class SynonymAttack(Attack):
    """Attack that replaces words with synonyms while preserving formatting."""

    def __init__(self, method="wordnet"):
        """
        Initialize the synonym attack with a specified method and swap probability.

        Args:
            method (str): The synonym replacement method to use. Options are:
                - "wordnet" (default): Uses WordNet for synonyms
                - "embedding": Uses word embeddings for similar words
                - "maskedlm": Uses masked language model for replacements
                - "hownet": Uses HowNet for synonyms
            probability (float): Probability of replacing a word with its synonym (0.0 to 1.0).
                               1.0 means replace all possible words (default)
                               0.0 means replace no words
                               0.5 means 50% chance to replace each word
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

        result = "".join(new_tokens)

        print("Debug synonym:")
        print(f"tokens:\n{tokens}\nnew_tokens:\n{result}")
        return result
