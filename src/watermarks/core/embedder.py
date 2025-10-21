from typing import List, Optional, Tuple
import torch
from tqdm import tqdm
from collections import defaultdict

from ..models.base import LanguageModel, BaseTokenizer
from ..prf.base import PRF
from ..perturb.base import PerturbFunction
from ..utils.helpers import get_keys_to_use, sample_key
from ..utils.debug import log_prf_output

class Embedder:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: BaseTokenizer,
        prf: PRF,
        perturb_fn: PerturbFunction,
        context_length: Optional[int] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prf = prf
        self.perturb_fn = perturb_fn
        self.context_length = context_length or model.context_length

    def embed(
        self,
        keys: List[bytes],
        h: List[str],
        m: List[int],
        delta: float,
        c: int,
        covertext_length: int,
    ) -> Tuple[str, dict, dict]:
        """
        Embed a message into generated text.
        
        Args:
            keys: List of keys, one per message bit
            h: History/context for generation
            m: Message bits to embed
            delta: Perturbation strength
            c: c-gram size for PRF
            covertext_length: Number of tokens to generate
        
        Returns:
            Tuple containing:
            - Generated text (str)
            - Token information (dict)
            - Key selection counters (dict) mapping key to number of times selected
        """
        # Initialize counter for each key
        key_counters = defaultdict(int)
        
        # get keys that correspond to 1s in message
        watermarking_keys = get_keys_to_use(m, keys)

        # if message is all 0s, can't sample watermarking key
        if len(watermarking_keys) == 0:
            return self._generate_without_watermark(h, covertext_length)

        # Compute the salt s from the history h
        s = len(h)

        # tokenize history
        text = ''.join(h)
        tokens = self.tokenizer(text, return_tensors='pt')

        # watermark for each key
        for j in tqdm(range(covertext_length)):
            sampled_key = sample_key(watermarking_keys)
            key_counters[sampled_key] += 1
            
            # Use only the last context_length tokens for prediction
            if tokens['input_ids'].shape[1] > self.context_length:
                context_tokens = {
                    'input_ids': tokens['input_ids'][:, -self.context_length:],
                    'attention_mask': tokens['attention_mask'][:, -self.context_length:]
                }
            else:
                context_tokens = tokens
                
            # Apply the language model over previous tokens to get a probability distribution
            p = self.model.get_next_token_distribution(context_tokens)
            # compute r
            r = self.prf(sampled_key, s, context_tokens, c)
            # perturb p
            p_prime = self.perturb_fn(p, r, delta)
            # sample next token with p_prime
            token = self.model.sample_token(p_prime, tokens['input_ids'][0][-1].item())
            token_tensor = torch.tensor([[token]])
            tokens['input_ids'] = torch.cat((tokens['input_ids'], token_tensor), dim=1)
            tokens['attention_mask'] = torch.cat((tokens['attention_mask'], torch.tensor([[1]])), dim=1)

        sampled_text = self.tokenizer.decode(tokens['input_ids'][0].tolist())
        return sampled_text, tokens, dict(key_counters)  # Convert defaultdict to regular dict for return

    def _generate_without_watermark(self, history: List[str], length: int) -> Tuple[str, dict, dict]:
        """Generate text without watermark when message is all zeros."""
        text = ''.join(history)
        tokens = self.tokenizer(text, return_tensors='pt')

        for _ in tqdm(range(length)):
            if tokens['input_ids'].shape[1] > self.context_length:
                context = {
                    'input_ids': tokens['input_ids'][:, -self.context_length:],
                    'attention_mask': tokens['attention_mask'][:, -self.context_length:]
                }
            else:
                context = tokens
                
            p = self.model.get_next_token_distribution(context)
            token = self.model.sample_token(p, tokens['input_ids'][0][-1].item())
            
            token_tensor = torch.tensor([[token]])
            tokens['input_ids'] = torch.cat((tokens['input_ids'], token_tensor), dim=1)
            tokens['attention_mask'] = torch.cat((tokens['attention_mask'], torch.tensor([[1]])), dim=1)

        sampled_text = self.tokenizer.decode(tokens['input_ids'][0].tolist())
        return sampled_text, tokens, {}  # Empty dict for no watermark case
