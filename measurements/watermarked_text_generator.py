import json
import sys
import random
from watermark import (
    GPT2Model,
    AESPRF,
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)

def generate_watermarked_text(id):
    n_bits = 3              # Length of message to hide
    epsilon = 0.05          # 95% success probability
    delta = 0.2             # Perturbation strength
    safety_factor = 10
    c = 5                   # n-gram length for PRF used by watermarking

    # Instantiate required components.
    calculator = SmoothCovertextCalculator()
    required_length = calculator.get_covertext_length(n=n_bits, epsilon=epsilon, delta=delta, safety_factor=safety_factor)

    model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size - 1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)

    # Watermarking parameters.
    message = [random.randint(0, 1) for _ in range(3)]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["How have you been?"]  # Context

    # Generate watermarked text.
    watermarked_text, tokens, _ = embedder.embed(keys=keys, h=history, m=message, delta=delta, c=c, covertext_length=required_length)

    # Convert tokens to a serializable format (list of token IDs)
    serializable_tokens = tokens['input_ids'].tolist()

    # Save to JSON file.
    output_data = {
        "id": id,
        "watermarked_text": watermarked_text,
        "tokens": serializable_tokens,
        "message": message
    }

    with open(f"watermarked_texts/{id}.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Generated watermarked text for ID: {id}, saved as {id}.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watermark_generator.py <id>")
        sys.exit(1)

    generate_watermarked_text(sys.argv[1])