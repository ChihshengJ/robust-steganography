from llm import get_next_word_distribution, get_next_token_distribution, sample_token, sample_token_id, tokenizer, model, vocab_size
from prf import PRF, PRF_t
from Helpers import get_keys_to_use, get_limit, sample_key
from tqdm import tqdm
import torch

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
  # print('watermarking_keys: ', watermarking_keys)
  # get loop limit that is large enough that in expectation, each watermarking key is used enough
  limit = get_limit(len(watermarking_keys))
  # Compute the salt s from the history h
  s = len(h)

  #! for now, concatenate the entire history together as the starting text
  text = ''.join(h) #! what seperator should we use?
  tokens = tokenizer(text, return_tensors='pt')
  # print('h: ', h)
  # print('text: ', text)

  # watermark for each key
  for j in tqdm(range(limit)):
    i = sample_key(watermarking_keys)
    # print('i: ', i)
    # Apply the language model over previous tokens to get a probability distribution p over the tth token
    p = get_next_token_distribution(tokens, model)
    # p = get_next_word_distribution(text, tokenizer, model)
    # print('p: ', p)
    # compute r
    #! should only feed in the c prior tokens, not all of text
    r = PRF_t(i, s, tokens, c)
    # print('i, s, text, c: ', i[0], s, text, c)
    # print('j, i, r: ', j, i[0], r)
    # print('r: ', r)
    p_prime = perturb(p, r, delta)
    # print('p_prime: ', p_prime)
    # sample next token with p_prime
    token = sample_token_id(p_prime)
    # with open("./encode.log", "a") as myfile:
    #   myfile.write('j: {j}\n'.format(j=j))
    #   myfile.write('i: {i}\n'.format(i=keys.index(i)))
    #   myfile.write('r: {r}\n'.format(r=r[:10]))
    #   myfile.write('token: {token}\n'.format(token=token))
    # print(tokens)
    # print(token)
    token_tensor = torch.tensor([[token]])
    tokens['input_ids'] = torch.cat((tokens['input_ids'], token_tensor), dim=1)
    tokens['attention_mask'] = torch.cat((tokens['attention_mask'], torch.tensor([[1]])), dim=1)
    # tokens.append(token_tensor)

#   print(tokens['input_ids'])
  text =[tokenizer.decode(t.item()) for t in tokens['input_ids'][0]]
  text = ''.join(text)

  return text, tokens

def decode(keys, h, ct, z, c):
  # Compute the salt s from the history h
  s = len(h)
  # initialize counters for each bit in m (seeing if the threshold is crossed for that bit)
  counters = [0 for _ in range(len(keys))]

  #! need to add tokens for h to beginning of tokens?
  # tokenize stegotext
  tokens = tokenizer(ct, return_tensors='pt') # this makes it line up with h as a starting point
#   print(tokens)
  # text needs to start at h (encode's text starts at h)
  # when making a back and forth system, one side will probably have to call this function with h -1
  # text = ''

  h_text = ''.join(h)
  h_tokens = tokenizer(h_text, return_tensors='pt')['input_ids'][0]
#   print(h_tokens)

  # for each token in stegotext
  for j in tqdm(range(len(h_tokens), len(tokens['input_ids'][0]))):
    # print('j: ', j)
    # text += tokenizer.decode(tokens[j])
    # test each key
    for i, key in enumerate(keys):
      # print('i: ', i)
    #   print(tokens[0:j])
      partial_tokens = {'input_ids': tokens['input_ids'][:, 0:j], 'attention_mask': tokens['attention_mask'][:, 0:j]}
    #   print(partial_tokens)
      r = PRF_t(key, s, partial_tokens, c)
    #   print('i, s, text, c: ', i, s, text, c)
    #   print('j, i, r: ', j, i, r)
      # print('r: ', r)
      current_token_index = tokens['input_ids'][0][j].item()
      # print('current_token_index: ', current_token_index)
      # print('r[current_token_index]: ', r[current_token_index])
      # For testing I need something that says: this is the next token. Here's what went into the PRF for sampling it.
    #   with open("./decode.log", "a") as myfile:
    #     myfile.write('j: {j}\n'.format(j=j))
    #     myfile.write('i: {i}\n'.format(i=i))
    #     myfile.write('r: {r}\n'.format(r=r[:10]))
    #     t = tokenizer.decode(tokens[j])
    #     myfile.write('token: {t}\n'.format(t=t))
      if (r[current_token_index] == 1):
        counters[i] += 1
        # print('counters: ', counters)
    # text += tokenizer.decode(tokens[j])

#   print(counters)

  return counters, tokens

