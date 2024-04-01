# make deterministic for testing
from config_determinism import set_deterministic_behavior
set_deterministic_behavior()

# rest of needed imports
from llm import tokenizer
from steg import encode, decode
from Helpers import detect, get_limit
import torch

token_match = 0

# results:
# [-0.4472135954999579, 1.073312629199899, 1.6099689437998486, 1.2521980673998823, 1.6994116628998401, -1.8782971010998233]
# use 0.001 for now
def find_delta(keys, h, m, c):
  deltas = [0.00001, 0.0001, .001, .01, .1, .5]
  delta_counters = [0 for _ in deltas]
  # search for what delta works best
  for _ in range(10):
    for delta in deltas:
      recovered_counters = test(keys, h, m, delta, c)
      result = recovered_counters[0]
      delta_counters[deltas.index(delta)] += result
  avg_delta_counters = [x / 10 for x in delta_counters]
  print(avg_delta_counters)
  detection = [detect(get_limit(None) * 10, x) for x in delta_counters]
  print(detection)

def mismatch(enc_ids, dec_ids):
  enc = [tokenizer.decode(t) for t in enc_ids]
  dec = [tokenizer.decode(t) for t in dec_ids]
  print(enc)
  print(dec)

  # compare how many indices are the same
  same = 0
  # smaller list index
  limit = min(len(enc_ids), len(dec_ids))
  for i in range(limit):
    if enc_ids[i] == dec_ids[i]:
      same += 1
    else:
        print('mismatch at index: ', i)
  print('num same: ', same)
  print('dec length: ', len(dec_ids))

def test(keys, h, m, delta, c):
  global token_match
  ct, tokens = encode(keys, h, m, delta, c)
  print('-------------------')
  recovered_counters, decode_tokens = decode(keys, h, ct, None, c)
  same = torch.equal(tokens['input_ids'], decode_tokens['input_ids'])
  print('Tokens match: ', same)
  if (same):
    token_match += 1
  if (not same):
    mismatch(tokens['input_ids'][0], decode_tokens['input_ids'][0])

  print('-------------------')
  print(ct)
  print(recovered_counters)

#   for i in range(len(recovered_counters)):
#     print(detect(get_limit(None), recovered_counters[i]))
  m_prime = [1 if detect(get_limit(None), x) else 0 for x in recovered_counters]
  print('m_prime: ', m_prime)

  return recovered_counters

m = [1, 0, 1, 1, 0]
l = len(m)
keys = [b'\0' * 32, b'\1' * 32, b'\2' * 32, b'\3' * 32, b'\4' * 32]
h = ['Hey! Want to get coffee? ', 'Would Friday work for you?']
c = 3
delta = 0.001

# find_delta(keys, h, m, c)

# delta = 0.01
# test how many times encode and decode perfectly match
# for _ in range(15):
#   test(keys, h, m, delta, c)

test(keys, h, m, delta, c)

print('Token match: ', token_match)