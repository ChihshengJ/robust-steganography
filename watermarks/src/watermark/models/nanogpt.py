import os
import torch
from .base import LanguageModel, BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    def __init__(self):
        self.chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self._vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text, return_tensors=None):
        ids = [self.stoi.get(c, self.stoi[' ']) for c in text]
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([ids]),
                'attention_mask': torch.ones(1, len(ids), dtype=torch.long)
            }
        return ids
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return ''.join(self.itos.get(i, ' ') for i in ids)

    def __call__(self, text, return_tensors=None):
        return self.encode(text, return_tensors=return_tensors)

    @property
    def vocab_size(self):
        return self._vocab_size


class NanoGPTModel(LanguageModel):
    def __init__(self, model_path="sosier/nanoGPT-shakespeare-char-tied-weights"):
        """Initialize NanoGPT model using HuggingFace's auto classes."""
        from transformers import AutoModel
        
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True  # Required for custom NanoGPT implementation
        )
        self._tokenizer = CharacterTokenizer()
        self._context_length = 250

    def get_next_token_distribution(self, input_tokens):
        if isinstance(input_tokens, dict) and 'input_ids' in input_tokens:
            idx = input_tokens['input_ids']
        else:
            idx = self._tokenizer.encode(input_tokens, return_tensors='pt')['input_ids']
        
        output = self._model(idx)
        logits = output[0][0]
        all_layers_probabilities = torch.softmax(logits, dim=-1)
        return all_layers_probabilities[-1]

    def sample_token(self, probabilities, prev_token=None):
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


# Global instances to exactly match llm.py
tokenizer = CharacterTokenizer()
model = NanoGPTModel()
vocab_size = tokenizer.vocab_size
max_token_id = vocab_size - 1
bpe = None

# Global functions to exactly match llm.py API
def get_next_word_distribution(input_text, tokenizer, model):
    encoded_input = tokenizer(input_text, return_tensors='pt')
    return get_next_token_distribution(encoded_input)

def get_next_token_distribution(input, model_unused=None):
    if isinstance(input, dict) and 'input_ids' in input:
        idx = input['input_ids']
    else:
        idx = tokenizer(input, return_tensors='pt')['input_ids']
    
    output = model(idx)
    logits = output[0][0]
    all_layers_probabilities = torch.softmax(logits, dim=-1)
    return all_layers_probabilities[-1]

def sample_token(probabilities):
    index = torch.multinomial(probabilities, 1).item()
    return tokenizer.decode(index)

def sample_token_id(probabilities, prev_token=None):
    return torch.multinomial(probabilities, 1).item()

def get_next_word(input_text, tokenizer, model):
    encoded_input = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_input)
    logits = output[0][0]
    all_layers_probabilities = torch.softmax(logits, dim=-1)
    probabilities = all_layers_probabilities[-1]
    index = torch.multinomial(probabilities, 1).item()
    return tokenizer.decode(index)

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(token_ids):
    return tokenizer.decode(token_ids)

def normalize_text(text):
    return ''.join(c if c in tokenizer.stoi else ' ' for c in text)
