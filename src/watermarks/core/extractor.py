from typing import List, Optional, Tuple
import torch
from tqdm import tqdm

from ..models.base import LanguageModel, BaseTokenizer
from ..prf.base import PRF
from ..utils.debug import log_prf_output

class Extractor:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: BaseTokenizer,
        prf: PRF,
        context_length: Optional[int] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prf = prf
        self.context_length = context_length or model.context_length

    def extract(
        self,
        keys: List[bytes],
        h: List[str],
        ct: str,
        c: int
    ) -> Tuple[List[int], dict]:
        """Extract watermark from covertext."""
        # Compute the salt s from the history h
        s = len(h)

        # Initialize counters for each key
        counters = [0 for _ in keys]

        # First tokenize history to know where watermarking starts
        history_text = ''.join(h)
        history_tokens = self.tokenizer(history_text, return_tensors='pt')
        history_len = history_tokens['input_ids'].shape[1]

        # Now tokenize full text (history + covertext)
        # full_text = history_text + ct
        full_text = ct
        tokens = self.tokenizer(full_text, return_tensors='pt')

        # For each position in covertext (after history)
        for j in tqdm(range(history_len, tokens['input_ids'].shape[1])):
            # Use only the last context_length tokens for PRF
            if j > self.context_length:
                context_tokens = {
                    'input_ids': tokens['input_ids'][:, j-self.context_length:j],
                    'attention_mask': tokens['attention_mask'][:, j-self.context_length:j]
                }
            else:
                context_tokens = {
                    'input_ids': tokens['input_ids'][:, :j],
                    'attention_mask': tokens['attention_mask'][:, :j]
                }

            # Check each key
            for i, key in enumerate(keys):
                # Get PRF output for this position and key
                r = self.prf(key, s, context_tokens, c)
                # If chosen token is marked 1 by PRF, increment counter
                if r[tokens['input_ids'][0][j].item()] == 1:
                    counters[i] += 1

        return counters, tokens
