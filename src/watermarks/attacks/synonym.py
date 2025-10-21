import re
from random import choice, random
from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import (
    WordSwapWordNet,
    WordSwapEmbedding,
    WordSwapMaskedLM,
    WordSwapHowNet
)

class SynonymAttack:
    """Attack that replaces words with synonyms while preserving formatting."""
    
    def __init__(self, method="wordnet", probability=1.0):
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
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
            
        self.method = method
        self.probability = probability
        
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

    def __call__(self, text: str) -> str:
        """Apply the synonym replacement attack."""
        if self.probability == 0:
            return text
            
        # Split text into words and whitespace, keeping both
        tokens = re.split(r'(\s+)', text)
        
        # Process only non-whitespace tokens
        new_tokens = []
        for token in tokens:
            if token.strip():  # If token is not whitespace
                # Randomly decide whether to try replacing this word
                if random() < self.probability:
                    augmented_texts = self.augmenter.augment(token)
                    if augmented_texts:
                        single_word_synonyms = [t for t in augmented_texts if len(t.split()) == 1]
                        if single_word_synonyms:
                            # Randomly select a synonym if available
                            new_tokens.append(choice(single_word_synonyms))
                            continue
                new_tokens.append(token)  # Keep original if no replacement or probability check fails
            else:
                new_tokens.append(token)  # Keep whitespace as is
                
        return ''.join(new_tokens)