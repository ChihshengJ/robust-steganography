import numpy as np
import math

def get_limit(num_watermarks):
  #! temporary constant
  return 150

def get_keys_to_use(m, keys):
  return [key for key, flag in zip(keys, m) if flag == 1]

def sample_key(keys):
  random_index = np.random.randint(len(keys))
  return keys[random_index]

# T is the number of tokens generated
# s_g is the number of green list tokens
def detect(T, s_g):
  print (T, s_g)
  z = (2 * (s_g - T / 2)) / math.sqrt(T)
  print(z)
  return z > 4