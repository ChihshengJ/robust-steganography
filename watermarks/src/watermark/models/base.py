from abc import ABC, abstractmethod
import torch

class LanguageModel(ABC):
    """Base class for language models."""
    
    @abstractmethod
    def get_next_token_distribution(self, input_tokens):
        """Get probability distribution for next token."""
        pass
    
    @abstractmethod
    def sample_token(self, probabilities, prev_token=None):
        """Sample next token from probability distribution."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self):
        """Return vocabulary size."""
        pass
    
    @property
    @abstractmethod
    def context_length(self):
        """Return maximum context length."""
        pass
    
    @property
    @abstractmethod
    def tokenizer(self):
        """Return the model's tokenizer."""
        pass


class BaseTokenizer(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text, return_tensors=None):
        """Convert text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids):
        """Convert token IDs to text."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self):
        """Return vocabulary size."""
        pass
