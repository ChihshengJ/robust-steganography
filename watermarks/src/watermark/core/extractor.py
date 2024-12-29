from typing import List, Optional, Tuple
import torch
from tqdm import tqdm

from ..models.base import LanguageModel, BaseTokenizer
from ..prf.base import PRF

class Extractor:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: BaseTokenizer,
        prf: PRF
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prf = prf

    def extract(
        self,
        keys: List[bytes],
        h: List[str],
        ct: str,
        c: int
    ) -> Tuple[List[int], dict]:
        """Matches decode from steg.py"""
        # Compute the salt s from the history h
        s = len(h)
        
        # initialize counters for each bit in m
        counters = [0 for _ in range(len(keys))]
        
        # tokenize stegotext and history
        text = ct
        tokens = {'input_ids': torch.tensor([self.tokenizer.encode(text)]), 
                 'attention_mask': torch.tensor([[1] * len(self.tokenizer.encode(text))])}
        
        h_text = ''.join(h)
        h_tokens = {'input_ids': torch.tensor([self.tokenizer.encode(h_text)]), 
                   'attention_mask': torch.tensor([[1] * len(self.tokenizer.encode(h_text))])}

        # for each token in stegotext after history
        for j in tqdm(range(len(h_tokens['input_ids'][0]), len(tokens['input_ids'][0]))):
            # test each key
            for i, key in enumerate(keys):
                partial_tokens = {
                    'input_ids': tokens['input_ids'][:, 0:j],
                    'attention_mask': tokens['attention_mask'][:, 0:j]
                }
                r = self.prf(key, s, partial_tokens, c)
                current_token_index = tokens['input_ids'][0][j].item()
                if r[current_token_index] == 1:
                    counters[i] += 1
                    
        return counters, tokens
