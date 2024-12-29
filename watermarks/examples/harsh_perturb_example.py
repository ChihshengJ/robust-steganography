"""
Example of watermarking using the harsh perturbation method.
This demonstrates more aggressive watermarking that may produce
less natural text but potentially stronger watermarks.
"""

from watermark import (
    NanoGPTModel,
    AESPRF,
    HarshPerturb,
    Embedder,
    Extractor,
    set_seed
)

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize with harsh perturbation
    model = NanoGPTModel()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = HarshPerturb()  # Using harsh perturbation instead of delta
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Setup watermarking parameters
    message = [1, 0, 1]
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]
    history = ["To be, or not to be- that is the question:"]
    c = 5  # Length of n-grams used by PRF for watermarking
    delta = 0.1  # Perturbation strength
    
    # Generate watermarked text
    print("Generating watermarked text with harsh perturbation...")
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