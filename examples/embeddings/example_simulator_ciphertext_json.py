from robust_steganography.core.steg_system import StegSystem
from robust_steganography.core.hash_functions import OracleHash
from robust_steganography.core.error_correction import ConvolutionalCode
from robust_steganography.core.encoder import CiphertextEncoder
from robust_steganography.config.system_prompts import TWO_WAY_DYNAMIC
from robust_steganography.core.simulation import Simulator
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import openai

# Generate encryption key and initialize AES
key = get_random_bytes(32)  # 256-bit key
cipher = AES.new(key, AES.MODE_EAX)

# Initialize components
client = None
hash_fn = OracleHash(output_length=4, error_rate=0.1)
ecc = ConvolutionalCode(block_size=hash_fn.get_output_length())
encoder = CiphertextEncoder(serialization_format='json')  # Using JSON format
system_prompt = TWO_WAY_DYNAMIC

system = StegSystem(
    client=client,
    hash_function=hash_fn,
    error_correction=ecc,
    encoder=encoder,
    system_prompt=system_prompt,
    chunk_length=50,
    simulator=Simulator()
)

# Encrypt and hide message
message = "abcdefghijklmnopqrstuvwxyz"
nonce = cipher.nonce
ciphertext = cipher.encrypt(message.encode('utf-8'))

# Create JSON-serializable dictionary
encryption_data = {
    'nonce': base64.b64encode(nonce).decode('utf-8'),
    'ciphertext': base64.b64encode(ciphertext).decode('utf-8')
}

history = [
    "What are you up to today?",
    "Nothing much, just working on a project.",
    "Want to grab coffee and discuss it?"
]

stego_texts = system.hide_message(encryption_data, history)
print("Stego texts:", stego_texts)

# Recover and decrypt message
recovered_data = system.recover_message(stego_texts)
recovered_nonce = base64.b64decode(recovered_data['nonce'])
recovered_ciphertext = base64.b64decode(recovered_data['ciphertext'])

# Decrypt
decrypt_cipher = AES.new(key, AES.MODE_EAX, nonce=recovered_nonce)
decrypted_message = decrypt_cipher.decrypt(recovered_ciphertext).decode('utf-8')
print("Recovered message:", decrypted_message) 