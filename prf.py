from bitstring import BitArray
import nacl.encoding
import nacl.hash
import nacl.secret
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import hashlib
import math

from llm import tokenizer, vocab_size, max_token_id

#! Need the output of the prf to be at least the vocab_size
def prf(key, data):
    iv = b'\0' * 16
    cipher = AES.new(key, AES.MODE_CBC, iv)
    prf_output = cipher.encrypt(pad(data, AES.block_size))

    extended_output = b''
    counter = 0
    while len(extended_output) < vocab_size:
        # Concatenate the PRF output with the counter and hash the result
        data_to_hash = prf_output + counter.to_bytes(8, 'big')  # 8 bytes for the counter
        hash_output = hashlib.sha256(data_to_hash).digest()
        extended_output += hash_output
        counter += 1
    return extended_output[:vocab_size]

def int_list_to_bytes(int_list):
    # determine the byte size to handle all token_ids
    byte_length = (max_token_id.bit_length() + 7) // 8  # Calculate byte length needed

    # Convert each integer to a byte sequence of the same length
    bytes_list = [i.to_bytes(byte_length, 'big') for i in int_list]

    # Concatenate all byte sequences into a single byte string
    byte_string = b''.join(bytes_list)

    return byte_string

# c is the number of tokens to use
def PRF(key, salt, n_gram, c):
  encoded_text = tokenizer(n_gram, return_tensors='pt')
  full_gram = encoded_text["input_ids"].tolist()[0]
  c_gram = full_gram[-c:]
  salted_bytes = [salt] + c_gram
  encoded_bytes = int_list_to_bytes(salted_bytes)

  digest = nacl.hash.sha256(encoded_bytes) # 64 bytes
  return truncate_to_vocab_size(prf(key, digest), vocab_size)
  # return ([1] * 25000) + ([0] * (vocab_size - 25000))

# c is the number of tokens to use
def PRF_t(key, salt, n_gram, c):
  full_gram = n_gram["input_ids"].tolist()[0]
  c_gram = full_gram[-c:]
  salted_bytes = [salt] + c_gram
  encoded_bytes = int_list_to_bytes(salted_bytes)

  digest = nacl.hash.sha256(encoded_bytes) # 64 bytes
  return truncate_to_vocab_size(prf(key, digest), vocab_size)
  # return ([1] * 25000) + ([0] * (vocab_size - 25000))

"""
Truncate the output of a PRF to the size of the vocabulary (1 bit per vocab word)
"""
def truncate_to_vocab_size(data, vocab_size):
  # turn bytes into bitstring
  bits = BitArray(data).bin
  bit_array = [int(b) for b in bits]

  # truncate to vocab size
  truncated_bit_array = bit_array[:vocab_size]

  return truncated_bit_array

# truncate_to_vocab_size(prf(digest), vocab_size)