import openai
from embeddings.core.encoder import MinimalEncoder
from embeddings.core.error_correction import RepetitionCode
from embeddings.core.hash_functions import RandomProjectionHash
from embeddings.core.steg_system import StegSystem

# Initialize components
client = openai.OpenAI()
hash_fn = RandomProjectionHash(embedding_dim=3072, num_bits=3)
# ecc = ConvolutionalCode(K=2)
ecc = RepetitionCode(repetitions=5)
system_prompt = 'You are a highly dynamic conversational model tasked with generating responses that are extremely varied in tone, content, and structure. Each response should aim to be unique and take the conversation in a new and unexpected direction. You can introduce sudden topic changes, challenge previous statements, or bring up something entirely unrelated. Embrace the unexpected: shift perspectives, introduce controversial ideas, or pose hypothetical questions. You can respond positively or negatively and DO NOT START RESPONSES with "Ah, {repeated information}" or anything similar. Avoid repeating any phrases or structures from previous responses. Your goal is to ensure each continuation is distinct, unpredictable, and creative.'

system = StegSystem(
    client=client,
    hash_function=hash_fn,
    error_correction=ecc,
    encoder=MinimalEncoder(),
    system_prompt=system_prompt,
    chunk_length=10,
)

# Hide message
message = "x"
history = [
    "What are you up to today?",
    "Nothing much, just working on a project.",
    "Want to grab coffee and discuss it?",
]

stego_texts = system.hide_message(message, history)
print("Stego texts:", stego_texts)
print("length:", len(stego_texts))

# Recover message
recovered = system.recover_message(stego_texts)
print("Recovered message:", recovered)
