import openai
from robust_steganography.config.system_prompts import CORPORATE_MONOLOGUE
from robust_steganography.core.encoder import MinimalEncoder
from robust_steganography.core.error_correction import RepetitionCode
from robust_steganography.core.hash_functions import PCAHash
from robust_steganography.core.steg_system import StegSystem
from robust_steganography.utils import load_pca_model

# Initialize components
client = openai.OpenAI()
pca_model_path = "../src/robust_steganography/models/pca_corporate.pkl"
pca_model = load_pca_model(pca_model_path)
hash_fn = PCAHash(pca_model=pca_model, start=0, end=1)
# ecc = ConvolutionalCode()
ecc = RepetitionCode(repetitions=5)
system_prompt = CORPORATE_MONOLOGUE

system = StegSystem(
    client=client,
    hash_function=hash_fn,
    error_correction=ecc,
    encoder=MinimalEncoder(),
    system_prompt=system_prompt,
    chunk_length=50,
)

# Hide message
message = "abcde"
history = [
    "I wanted to follow up regarding the implementation timeline for the new risk management system. Based on our initial assessment, we'll need to coordinate closely with both IT and Operations to ensure a smooth transition. Please review the attached documentation when you have a moment.",
    "After consulting with the development team, we've identified several key milestones that need to be addressed before proceeding. The current testing phase has revealed some potential integration issues with our legacy systems, particularly in the trade validation module. We're working on implementing the necessary fixes and expect to have an updated timeline by end of week.",
    "Given the complexity of these changes, I believe it would be beneficial to schedule a stakeholder review meeting. We should include representatives from Risk Management, IT Operations, and the Trading desk to ensure all requirements are being met. I've asked Sarah to coordinate calendars for next Tuesday afternoon.",
]

stego_texts = system.hide_message(message, history)
print("Stego texts:", stego_texts)

# Recover message
recovered = system.recover_message(stego_texts)
print("Recovered message:", recovered)
