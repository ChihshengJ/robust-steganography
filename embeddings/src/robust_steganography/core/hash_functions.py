from functools import partial
import numpy as np
from typing import Dict, Optional

class HashFunction:
    def __init__(self):
        pass
    
    def __call__(self, emb):
        raise NotImplementedError("Hash function must implement __call__")
        
    def get_output_length(self):
        raise NotImplementedError("Hash function must implement get_output_length")

class RandomProjectionHash(HashFunction):
    def __init__(self, embedding_dim=3072, num_bits=1):
        super().__init__()
        self.rand_matrix = np.random.randn(embedding_dim, num_bits)
        self.output_length = num_bits
        
    def __call__(self, emb):
        projection = emb @ self.rand_matrix
        hashes = (projection > 0).astype(int)
        return hashes.ravel()[0]
        
    def get_output_length(self):
        return self.output_length

class PCAHash(HashFunction):
    def __init__(self, pca_model, start=0, end=1):
        super().__init__()
        self.pca = pca_model
        self.start = start
        self.end = end
        self.output_length = end - start
        
    def __call__(self, emb):
        transformed = self.pca.transform(emb.reshape(1, -1))
        return (transformed[:, self.start:self.end] > 0).astype(int).ravel()[0]
        
    def get_output_length(self):
        return self.output_length

class OracleHash(HashFunction):
    def __init__(self, output_length: int, error_rate: float = 0.0, seed: Optional[int] = None):
        super().__init__()
        self.output_length = output_length
        self.error_rate = error_rate
        self.hash_memory: Dict[str, np.ndarray] = {}
        if seed is not None:
            np.random.seed(seed)
    
    def __call__(self, emb, corrupt: bool = False) -> np.ndarray:
        # Use embedding as key (convert to string for dict key)
        key = str(emb.tolist())
        
        # If we haven't seen this embedding before, generate random bits
        if key not in self.hash_memory:
            self.hash_memory[key] = np.random.randint(0, 2, self.output_length)
        
        bits = self.hash_memory[key]
        
        # Apply corruption during retrieval if requested
        if corrupt and self.error_rate > 0:
            mask = np.random.random(bits.shape) < self.error_rate
            bits = np.logical_xor(bits, mask).astype(int)
        return bits
    
    def get_output_length(self):
        return self.output_length
    