import hmac
import hashlib
import torch
import os
from typing import Dict, List
import numpy as np

from .base import PRF

class HMACPRF(PRF):
    """
    PRF implementation using HMAC with SHA-256.
    
    This implementation:
    1. Uses HMAC-SHA256 as the core cryptographic primitive
    2. Generates a binary sequence for each token based on:
       - The secret key
       - The salt (derived from history length)
       - The context (last n tokens)
       - The token index being considered
    3. Maps each token to 0 or 1 using the HMAC output
    """
    
    def __init__(self, vocab_size: int, max_token_id: int):
        """
        Initialize HMAC PRF.
        
        Parameters
        ----------
        vocab_size : int
            Size of the model's vocabulary
        max_token_id : int
            Maximum token ID in the vocabulary
        """
        self.vocab_size = vocab_size
        self.max_token_id = max_token_id
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a random 32-byte key suitable for HMAC-SHA256.
        
        Returns
        -------
        bytes
            A 32-byte random key
        """
        return os.urandom(32)
        
    def __call__(
        self,
        key: bytes,
        salt: int,
        context: Dict[str, torch.Tensor],
        c: int
    ) -> List[int]:
        """
        Generate binary labels for each possible next token.
        
        Parameters
        ----------
        key : bytes
            Secret key for HMAC
        salt : int
            Salt value (typically length of history)
        context : Dict[str, torch.Tensor]
            Context tokens used for prediction
        c : int
            Length of n-grams to consider
        
        Returns
        -------
        List[int]
            Binary labels for each token in vocabulary
        """
        # Convert context to list of token IDs
        context_tokens = context['input_ids'][0].tolist()[-c:]
        
        # Create HMAC instance
        h = hmac.new(key, digestmod=hashlib.sha256)
        
        # Update HMAC with salt and context
        h.update(salt.to_bytes(8, 'big'))
        for token in context_tokens:
            h.update(token.to_bytes(4, 'big'))
            
        # Generate labels for all possible next tokens
        labels = []
        for token_id in range(self.vocab_size):
            # Create new HMAC instance for each token
            h_token = h.copy()
            h_token.update(token_id.to_bytes(4, 'big'))
            
            # Use first bit of hash as label
            digest = h_token.digest()
            label = digest[0] & 1  # Get least significant bit
            labels.append(label)
            
        return labels 