"""
Example of watermarking with GPT-2.

Note: GPT-2 uses Byte-Pair Encoding (BPE) for tokenization, which means tokens
can be subwords or multiple words. Due to BPE merge rules, the same text can sometimes
be tokenized differently during encoding vs decoding. These tokenization mismatches
act similar to a small attack on the watermark, as they can disrupt the n-gram patterns
used for watermarking. This makes the watermarking less reliable but allows for more
natural language generation.
"""

from watermark import (
    GPT2Model,
    AESPRF,
    DeltaPerturb,
    Embedder,
    Extractor,
    set_seed
)

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize components
    model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = DeltaPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Setup watermarking parameters
    message = [1, 0, 1]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["The future of artificial intelligence is", "a topic of great debate."]  # Modern context
    c = 5  # Length of n-grams used by PRF for watermarking
    delta = 0.1  # Perturbation strength
    
    # Generate watermarked text
    print("Generating watermarked text using GPT-2...")
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