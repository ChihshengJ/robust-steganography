from robust_steganography.config.system_prompts import (
    TWO_WAY_DYNAMIC,
)
from robust_steganography.core.encoder import MinimalEncoder
from robust_steganography.core.error_correction import ConvolutionalCode
from robust_steganography.core.hash_functions import OracleHash
from robust_steganography.core.simulation import Simulator
from robust_steganography.core.steg_system import StegSystem

# Initialize components
# client = openai.OpenAI()
client = None
hash_fn = OracleHash(output_length=4, error_rate=0.1)
# ecc = RepetitionCode(repetitions=3, block_size=hash_fn.get_output_length())
ecc = ConvolutionalCode(block_size=hash_fn.get_output_length())
system_prompt = TWO_WAY_DYNAMIC

system = StegSystem(
    client=client,
    hash_function=hash_fn,
    error_correction=ecc,
    encoder=MinimalEncoder(),
    system_prompt=system_prompt,
    chunk_length=5,
    simulator=Simulator()  # Pass simulator instance directly
)

# Hide message
message = "mission compromised"
history = [
    "What are you up to today?",
    "Nothing much, just working on a project.",
    "Want to grab coffee and discuss it?"
]

stego_texts = system.hide_message(message, history)
print("Stego texts:", stego_texts)

# Recover message
recovered = system.recover_message(stego_texts)
print("Recovered message:", recovered) 
