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
#   print('p: ', p)
#   print('p sum: ', sum(p))
  # print('in perturb')
  #! check that this is right and that p can be updated in place. Also check that p' sums to 1
  N = vocab_size

  # Build I: set of indices in [N] for which p_i ∈ [2δ, 1 − 2δ].
  I = set()
  for i, p_i in enumerate(p):
    #! temp testing
    # I.add(i)
    #! end temp testing
    if (p_i >= 2 * delta and p_i <= 1 - 2 * delta):
      I.add(i)

  # Set w to be the number of indices in [N] for which r_i = 1 and δ′ = δw/(N′ − w).
  #! double check
  w = sum(r)
  #! the formula says N_prime - w. Is this a typo? N_prime is never defined
  delta_prime = (delta * w) / (N - w)

#   print('I: ', I)

  # Adjust probabilities
  for j in I:
    # print('j: ', j)
    # print('r[j]: ', r[j])
    # print('p[j]: ', p[j])
    if (r[j] == 1):
      p[j] += delta
    else:
      p[j] -= delta_prime
      #! temp testing
    #   if (p[j] < 0):
    #     p[j] = 0
      #! end temp testing
    # print('p_prime[j]: ', p[j])
  # the j not in I stay the same and since p was updated in place this has been handled
#   print('p\': ', p)
  p = p / sum(p)
#   print('p\' sum: ', sum(p))

  return p

"""
keys, history, message to be encoded, hardness parameter delta
"""
def encode(keys, h, m, delta, c):
  # get keys that correspond to 1s in message
  watermarking_keys = get_keys_to_use(m, keys)
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
    token = sample_token_id(p_prime)
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

