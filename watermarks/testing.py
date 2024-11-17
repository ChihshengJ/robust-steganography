# make deterministic for testing
from config_determinism import set_deterministic_behavior
set_deterministic_behavior()

# rest of needed imports
from llm import tokenizer
from steg import encode, decode
from prf import PRF_t
from Helpers import detect, get_z, get_limit, generate_n_grams_with_counts, count_maintained_n_grams_with_frequency, sample_bit
import torch
import os

token_match = 0

# results:
# [-0.4472135954999579, 1.073312629199899, 1.6099689437998486, 1.2521980673998823, 1.6994116628998401, -1.8782971010998233]
# [0.5059644256269407, 2.4033310217279684, 1.5178932768808222, 3.1622776601683795, 3.0357865537616444, 0.6957010852370434] for new version
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
  detection = [get_z(get_limit(None) * 10, x) for x in delta_counters]
  print(detection)

def mismatch(enc_ids, dec_ids, c):
  enc = [tokenizer.decode(t) for t in enc_ids]
  dec = [tokenizer.decode(t) for t in dec_ids]
  print(enc)
  print(dec)

  total_c_grams = sum(generate_n_grams_with_counts(enc, c).values())
  print('Total enc c-grams: ', total_c_grams)
  maintained_c_grams = count_maintained_n_grams_with_frequency(enc, dec, c)
  print('Maintained c-grams: ', maintained_c_grams)

def test(keys, h, m, delta, c):
  ct, tokens = encode(keys, h, m, delta, c)
  recovered_counters, decode_tokens = decode(keys, h, ct, None, c)
  m_prime = [1 if detect(get_limit(None), x) else 0 for x in recovered_counters]

  return m_prime

def test_with_logging(keys, h, m, delta, c):
  global token_match
  ct, tokens = encode(keys, h, m, delta, c)
  print('-------------------')
  recovered_counters, decode_tokens = decode(keys, h, ct, None, c)
  same = torch.equal(tokens['input_ids'], decode_tokens['input_ids'])
  print('Tokens match: ', same)
  if (same):
    token_match += 1
  if (not same):
    mismatch(tokens['input_ids'][0], decode_tokens['input_ids'][0], c)

  print('-------------------')
  print(ct)
  print(recovered_counters)

#   for i in range(len(recovered_counters)):
#     print(detect(get_limit(None), recovered_counters[i]))
  print([get_z(get_limit(None), x) for x in recovered_counters])
  m_prime = [1 if detect(get_limit(None), x) else 0 for x in recovered_counters]
  print('m: ', m)
  print('m_prime: ', m_prime)

  return m_prime

def batch_test(message_length, num_tests):
  print('Batch test')
  print('Covertext Size: ', get_limit(None))
  print('message_length: ', message_length)
  print('num_tests: ', num_tests)
  print('-------------------')
  matches = 0
  for _ in range(num_tests):
    m = [sample_bit() for _ in range(message_length)]
    keys = [os.urandom(32) for _ in range(message_length)]
    h = ['Hey! Want to get coffee? ', 'Would Friday work for you?']
    c = 3
    delta = 0.01
    m_prime = test(keys, h, m, delta, c)
    print('m: ', m)
    print('m_prime: ', m_prime)
    print(m == m_prime)
    print('-------------------')
    if m == m_prime:
      matches += 1
  print('Matches: ', matches)

def count_perfect_matches(keys, h, m, delta, c, num):
  # test how many times encode and decode perfectly match
  for _ in range(num):
    test(keys, h, m, delta, c)
  print('Token match: ', token_match)

def prf_bias():
  key = b'\0' * 32
  n_gram = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6]])}
  c = 3

  summed_percentages = 0

  for salt in range(100):
    output = PRF_t(key, salt, n_gram, c)
    percentage = sum(output) / len(output)
    summed_percentages += percentage
  
  print(summed_percentages / 100)

# m = [1, 0, 1, 1, 0]
m = [1, 0, 1]
l = len(m)
# keys = [b'\0' * 32, b'\1' * 32, b'\2' * 32, b'\3' * 32, b'\4' * 32]
keys = [b'\0' * 32, b'\1' * 32, b'\2' * 32,]
h = ['Hey! Want to get coffee? ', 'Would Friday work for you?']
c = 3
delta = 0.01

# test(keys, h, m, delta, c)
batch_test(6, 20)

