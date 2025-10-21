import numpy as np
from typing import Dict, List, Tuple

class Simulator:
    def __init__(self, embedding_dim=3072):
        self.embedding_dim = embedding_dim
        self.text_to_embedding: Dict[str, np.ndarray] = {}
        
    def generate_dummy_text(self, length=100) -> str:
        text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), size=length))
        if text not in self.text_to_embedding:
            self.text_to_embedding[text] = self.generate_dummy_embedding()
        return text
        
    def generate_dummy_embedding(self) -> np.ndarray:
        vec = np.random.randn(self.embedding_dim)
        return vec / np.linalg.norm(vec)
        
    def get_embedding(self, text: str) -> np.ndarray:
        return self.text_to_embedding[text]