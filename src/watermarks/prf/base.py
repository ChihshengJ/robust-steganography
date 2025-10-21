from abc import ABC, abstractmethod

class PRF(ABC):
    """Base class for pseudorandom functions."""
    
    @abstractmethod
    def __call__(self, key, salt, n_gram, context_length):
        """
        Generate pseudorandom bits for given input.
        
        Args:
            key: PRF key
            salt: Integer salt value
            n_gram: Token context dictionary with 'input_ids'
            context_length: Number of context tokens to use
            
        Returns:
            List of binary values with length equal to vocab size
        """
        pass
    
    @abstractmethod
    def generate_key(self):
        """Generate a new random key."""
        pass
