from llm import get_next_word_distribution, get_next_token_distribution, sample_token, sample_token_id, tokenizer, model, vocab_size
from prf import PRF, PRF_t
from Helpers import get_keys_to_use, get_limit, sample_key
from tqdm import tqdm
import torch

def prf_log(j, i, r, t, file):
  with open(file, "a") as myfile:
    myfile.write('j: {j}\n'.format(j=j))
    myfile.write('i: {i}\n'.format(i=i))
    myfile.write('r: {r}\n'.format(r=r[:10]))
    myfile.write('token: {token}\n'.format(token=t))

"""
Implements the perturb function from our paper.
p: distribution, r: prf output, delta: amount to adjust logits by
Returns a perturbed distribution p' (p is updated in place to become p' and p is returned)
"""
def perturb(p, r, delta):
  N = vocab_size

  # Set w to be the number of indices in [N] for which r_i = 1
  w = sum(r)

  # Build I: set of indices in [N] for which p_i ∈ [2δ, 1 − 2δ].
  I = set()
  for i, p_i in enumerate(p):
    if (p_i >= delta and p_i <= 1 - delta):
      I.add(i)

  num_inc = 0

  # Adjust probabilities
  for j in I:
    if (r[j] == 1):
      num_inc += 1
      p[j] += delta

  for j in range(len(r)):
    if (r[j] == 0):
      p[j] = 0

  # the j not in I stay the same and since p was updated in place this has been handled
  # if len(I) > 0:
  #   print(num_inc / len(I))

  p = p / sum(p)

  return p

def encode_zeros(h):
  limit = get_limit(None)

  # tokenize history
  text = ''.join(h) #! what seperator should we use?
  tokens = tokenizer(text, return_tensors='pt')

  for j in tqdm(range(limit)):
    # Apply the language model over previous tokens to get a probability distribution p over the tth token
    p = get_next_token_distribution(tokens, model)
    # sample next token
    token = sample_token_id(p, tokens['input_ids'][0][-1].item())
    token_tensor = torch.tensor([[token]])
    tokens['input_ids'] = torch.cat((tokens['input_ids'], token_tensor), dim=1)
    tokens['attention_mask'] = torch.cat((tokens['attention_mask'], torch.tensor([[1]])), dim=1)

  text =[tokenizer.decode(t.item()) for t in tokens['input_ids'][0]]
  text = ''.join(text)

  return text, tokens

"""
keys, history, message to be encoded, hardness parameter delta
"""
def encode(keys, h, m, delta, c):
  # get keys that correspond to 1s in message
  watermarking_keys = get_keys_to_use(m, keys)

  # if message is all 0s, can't sample watermarking key
  if len(watermarking_keys) == 0:
    return encode_zeros(h)

  # get loop limit that is large enough that in expectation, each watermarking key is used enough
  limit = get_limit(len(watermarking_keys))
  # Compute the salt s from the history h
  s = len(h)

  # tokenize history
  text = ''.join(h) #! what seperator should we use?
  tokens = tokenizer(text, return_tensors='pt')

  # watermark for each key
  for j in tqdm(range(limit)):
    i = sample_key(watermarking_keys)
    # Apply the language model over previous tokens to get a probability distribution p over the tth token
    p = get_next_token_distribution(tokens, model)
    # compute r
    r = PRF_t(i, s, tokens, c)
    p_prime = perturb(p, r, delta)
    # sample next token with p_prime
    token = sample_token_id(p_prime, tokens['input_ids'][0][-1].item())
    # prf_log(j, keys.index(i), r, token, "./encode.log")
    token_tensor = torch.tensor([[token]])
    tokens['input_ids'] = torch.cat((tokens['input_ids'], token_tensor), dim=1)
    tokens['attention_mask'] = torch.cat((tokens['attention_mask'], torch.tensor([[1]])), dim=1)

  text =[tokenizer.decode(t.item()) for t in tokens['input_ids'][0]]
  text = ''.join(text)

  return text, tokens

def decode(keys, h, ct, z, c):
  # Compute the salt s from the history h
  s = len(h)
  # initialize counters for each bit in m (seeing if the threshold is crossed for that bit)
  counters = [0 for _ in range(len(keys))]
  # tokenize stegotext
  tokens = tokenizer(ct, return_tensors='pt') # this makes it line up with h as a starting point

  # text needs to start at h (encode's text starts at h)
  h_text = ''.join(h)
  h_tokens = tokenizer(h_text, return_tensors='pt')['input_ids'][0]

  # for each token in stegotext
  for j in tqdm(range(len(h_tokens), len(tokens['input_ids'][0]))):
    # test each key
    for i, key in enumerate(keys):
      partial_tokens = {'input_ids': tokens['input_ids'][:, 0:j], 'attention_mask': tokens['attention_mask'][:, 0:j]}
      r = PRF_t(key, s, partial_tokens, c)
      current_token_index = tokens['input_ids'][0][j].item()
    #   prf_log(j - len(h_tokens), i, r, current_token_index, "./decode.log")
      if (r[current_token_index] == 1):
        counters[i] += 1

  return counters, tokens

