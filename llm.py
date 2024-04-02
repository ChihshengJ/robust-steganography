from transformers import GPT2Tokenizer, GPT2LMHeadModel #GPT2Model
import torch
import numpy as np

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
# Get the maximum token ID
vocab_size = tokenizer.vocab_size
max_token_id = vocab_size - 1

"""
Generate the next word distribution given an input text using a language model.

    Args:
        input_text (str): The input text for which the next word prediction is generated.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text encoding and decoding.
        model (transformers.PreTrainedModel): The language model used for prediction.

    Returns:
        list: The probability distribution of the next word based on the input text.

    This function takes an input text and utilizes a pre-trained language model to predict the most likely
    next word in the sequence. It encodes the input text, calculates the probabilities of the next word
    based on the language model's output, and returns the probability distribution of the next word.
"""
def get_next_word_distribution(input_text, tokenizer, model):
  # Encode the text and get output logits
  encoded_input = tokenizer(input_text, return_tensors='pt')
  output = model(**encoded_input)

  logits = output.logits[0]

  # Apply softmax to get probabilities
  all_layers_probabilities = torch.softmax(logits, dim=-1)

  # Get the probabilities from the last layer
  probabilities = all_layers_probabilities[-1]

  return probabilities

def get_next_token_distribution(input, model):
  # get output logits
  output = model(**input)

  logits = output.logits[0]

  # Apply softmax to get probabilities
  all_layers_probabilities = torch.softmax(logits, dim=-1)

  # Get the probabilities from the last layer
  probabilities = all_layers_probabilities[-1]

  return probabilities

def sample_token(probabilities):
  # Sample a token from the probability distribution
  index = torch.multinomial(probabilities, 1).item()
  #print(index, probabilities[index])

  # Decode the token IDs to text
  decoded_text = tokenizer.decode(index)
  return decoded_text

def sample_token_id(probabilities):
  # Sample a token from the probability distribution
  index = torch.multinomial(probabilities, 1).item()
  
  return index

"""
Generate the next word prediction given an input text using a language model.

    Args:
        input_text (str): The input text for which the next word prediction is generated.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text encoding and decoding.
        model (transformers.PreTrainedModel): The language model used for prediction.

    Returns:
        str: The predicted next word based on the input text.

    This function takes an input text and utilizes a pre-trained language model to predict the most likely
    next word in the sequence. It encodes the input text, calculates the probabilities of the next word
    based on the language model's output, and returns the predicted word as a string.
"""
def get_next_word(input_text, tokenizer, model):
  # Encode the text and get output logits
  encoded_input = tokenizer(input_text, return_tensors='pt')
  output = model(**encoded_input)

  logits = output.logits[0]

  # Apply softmax to get probabilities
  all_layers_probabilities = torch.softmax(logits, dim=-1)

  # Get the probabilities from the last layer
  probabilities = all_layers_probabilities[-1]

  #! this is where we would perturb the distribution

  # Sample a token from the probability distribution
  index = torch.multinomial(probabilities, 1).item()
  #print(index, probabilities[index])

  # Decode the token IDs to text
  decoded_text = tokenizer.decode(index)
  return decoded_text