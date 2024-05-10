import json
import os
import base64
from steg import encode, decode
from Helpers import detect, get_limit, sample_bit

def gen(keys, h, m, delta, c):
  ct, tokens = encode(keys, h, m, delta, c)
  recovered_counters, decode_tokens = decode(keys, h, ct, None, c)
  m_prime = [1 if detect(get_limit(None), x) else 0 for x in recovered_counters]
  encoded_keys = [base64.b64encode(key).decode('ascii') for key in keys]

  # if successful (m == m') log example to json object
  if m == m_prime:
    with open('data.json', 'r') as f:
      data = json.load(f)
      print(data)
      data.append({
        'keys': encoded_keys,
        'h': h,
        'm': m,
        'delta': delta, 
        'c': c,
        'limit': get_limit(None),
        'ct': ct,
      })
      with open('data.json', 'w') as f:
        json.dump(data, f)

for _ in range(100):
  message_length = 3
  m = [sample_bit() for _ in range(message_length)]
  keys = [os.urandom(32) for _ in range(message_length)]
  h = ['Hey! Want to get coffee? ', 'Would Friday work for you?']
  c = 3
  delta = 0.01

  gen(keys, h, m, delta, c)
