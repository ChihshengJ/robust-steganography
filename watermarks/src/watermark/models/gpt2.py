import torch
from transformers import GPT2Tokenizer as HFTokenizer, GPT2LMHeadModel
from .base import LanguageModel, BaseTokenizer

class GPT2Tokenizer(BaseTokenizer):
    """Wrapper for HuggingFace GPT2 tokenizer."""
    
    def __init__(self, model_name='gpt2-medium'):
        self._tokenizer = HFTokenizer.from_pretrained(model_name)
        self._vocab_size = self._tokenizer.vocab_size
        
    def encode(self, text: str, return_tensors=None):
        """Encode text to token IDs."""
        if return_tensors == 'pt':
            return self._tokenizer(text, return_tensors='pt')
        return self._tokenizer.encode(text)
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids)
        
    def __call__(self, text, return_tensors=None):
        return self.encode(text, return_tensors=return_tensors)
    
    @property
    def vocab_size(self):
        return self._vocab_size


class GPT2Model(LanguageModel):
    """Wrapper for HuggingFace GPT2 model."""
    
    def __init__(self, model_name='gpt2-medium'):
        self._model = GPT2LMHeadModel.from_pretrained(model_name)
        self._tokenizer = GPT2Tokenizer(model_name)
        self._context_length = self._model.config.n_ctx
        
    def get_next_token_distribution(self, input_tokens):
        """Get probability distribution for next token."""
        output = self._model(**input_tokens)
        logits = output.logits[0]
        all_layers_probabilities = torch.softmax(logits, dim=-1)
        return all_layers_probabilities[-1]
    
    def sample_token(self, probabilities, prev_token=None):
        """Sample next token from probability distribution."""
        return torch.multinomial(probabilities, 1).item()
    
    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size
    
    @property
    def context_length(self):
        return self._context_length
        
    @property
    def tokenizer(self):
        return self._tokenizer 