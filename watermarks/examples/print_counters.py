"""
Prints the key selection counters for a given message and model to compare to theory.
"""

from watermark import (
    ShakespeareNanoGPTModel,
    HMACPRF,
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)
from watermark.utils import detect

def main():
    # Setup parameters
    n_bits = 3  # Length of message to hide
    epsilon = 0.05  # 95% success probability
    delta = 0.2  # Perturbation strength
    safety_factor = 10
    
    # Calculate required covertext length
    calculator = SmoothCovertextCalculator()
    required_length = calculator.get_covertext_length(
        n=n_bits,
        epsilon=epsilon,
        delta=delta,
        safety_factor=safety_factor
    )
    print(f"\nRequired covertext length for {n_bits} bits with {(1-epsilon)*100}% accuracy: {required_length} tokens")
    
    # Initialize components
    model = ShakespeareNanoGPTModel()
    prf = HMACPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Setup watermarking parameters
    message = [1, 0, 1]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["To be, or not to be- that is the question:"]  # Shakespeare-style context
    c = 5  # Length of n-grams used by PRF for watermarking
    
    # Generate watermarked text of required length
    print("\nGenerating watermarked text of required length...")
    watermarked_text, _, key_counters = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,
        covertext_length=required_length  # Use calculated length
    )
    
    print(key_counters)

if __name__ == "__main__":
    main() 