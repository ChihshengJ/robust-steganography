import nacl.encoding
import nacl.hash
import nacl.secret
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import hashlib
from bitstring import BitArray

from .base import PRF

class AESPRF(PRF):
    """AES-based PRF implementation."""
    
    def __init__(self, vocab_size: int, max_token_id: int):
        """
        Initialize PRF with vocabulary parameters.
        
        Args:
            vocab_size: Size of the model's vocabulary
            max_token_id: Maximum token ID in the vocabulary
        """
        self.vocab_size = vocab_size
        self.max_token_id = max_token_id

    def __call__(self, key, salt, n_gram, c):
        """
        Generate pseudorandom bits using AES.
        
        Args:
            key: PRF key
            salt: Integer salt value
            n_gram: Token context dictionary with 'input_ids'
            c: Length of n-grams to use for watermarking
            
        Returns:
            List of binary values with length equal to vocab size
        """
        full_gram = n_gram["input_ids"].tolist()[0]
        c_gram = full_gram[-c:]  # Take last c tokens for n-gram
        salted_bytes = [salt] + c_gram
        encoded_bytes = self._int_list_to_bytes(salted_bytes)

        digest = nacl.hash.sha256(encoded_bytes)  # 64 bytes
        return self._truncate_to_vocab_size(self._prf(key, digest))

    def generate_key(self):
        """Generate a new 32-byte random key."""
        return get_random_bytes(32)

    def _int_list_to_bytes(self, int_list):
        """Convert list of integers to bytes."""
        # determine the byte size to handle all token_ids
        byte_length = (self.max_token_id.bit_length() + 7) // 8  # Calculate byte length needed

        # Convert each integer to a byte sequence of the same length
        bytes_list = [i.to_bytes(byte_length, 'big') for i in int_list]

        # Concatenate all byte sequences into a single byte string
        return b''.join(bytes_list)

    def _prf(self, key, data):
        """Original prf function implementation"""
        iv = b'\0' * 16
        cipher = AES.new(key, AES.MODE_CBC, iv)
        prf_output = cipher.encrypt(pad(data, AES.block_size))

        extended_output = b''
        counter = 0
        while len(extended_output) < self.vocab_size:
            # Concatenate the PRF output with the counter and hash the result
            data_to_hash = prf_output + counter.to_bytes(8, 'big')  # 8 bytes for the counter
            hash_output = hashlib.sha256(data_to_hash).digest()
            extended_output += hash_output
            counter += 1
        return extended_output[:self.vocab_size]

    def _truncate_to_vocab_size(self, data):
        """Original truncate_to_vocab_size implementation"""
        # turn bytes into bitstring
        bits = BitArray(data).bin
        bit_array = [int(b) for b in bits]

        # truncate to vocab size
        truncated_bit_array = bit_array[:self.vocab_size]

        return truncated_bit_array
