"""
Example of watermarking with the Shakespeare NanoGPT model.

Note 1: This is a character-level model, meaning each token is a single character.
Because of this, token encode/decode operations will never have mismatches - 
each character maps to exactly one token and vice versa. This makes the watermarking
more reliable but limits the model to character-by-character generation.

Note 2: This example uses hardcoded small values for covertext_length and detection
threshold to demonstrate basic usage of the Shakespeare model. For proper message
recovery with statistical guarantees, see covertext_length_example.py which shows
how to calculate the appropriate covertext length and detection parameters.
"""

from watermark import (
    ShakespeareNanoGPTModel,
    AESPRF,
    SmoothPerturb,
    Embedder,
    Extractor,
    set_seed
)

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize components
    model = ShakespeareNanoGPTModel()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Setup watermarking parameters
    message = [1, 0, 1]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["To be, or not to be- that is the question:"]  # Shakespeare-style context
    c = 5  # Length of n-grams used by PRF for watermarking
    delta = 0.1  # Perturbation strength
    
    # Generate watermarked text
    print("Generating watermarked Shakespeare-style text...")
    watermarked_text, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,  # n-gram length for PRF
        covertext_length=100
    )
    
    print("\nGenerated text:")
    print(watermarked_text)
    
    # Extract watermark
    print("\nExtracting watermark...")
    recovered_counters, _ = extractor.extract(
        keys=keys,
        h=history,
        ct=watermarked_text,
        c=c  # Must use same n-gram length as embedding
    )
    
    # Detect watermark bits
    recovered_message = [1 if counter > 50 else 0 for counter in recovered_counters]
    
    print("\nResults:")
    print(f"Original message: {message}")
    print(f"Recovered message: {recovered_message}")
    print(f"Success: {message == recovered_message}")

if __name__ == "__main__":
    main() 