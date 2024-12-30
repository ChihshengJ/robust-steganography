"""
Example demonstrating how to calculate and use the required covertext length
for reliable message recovery.

This example:
1. Uses the covertext length calculator to determine how long the generated text
   needs to be to recover a 3-bit message with 95% accuracy
2. Generates watermarked text of that length using the Shakespeare model
3. Extracts the watermark to verify successful recovery

The covertext length calculation accounts for:
- Message length (3 bits)
- Desired accuracy (95% = epsilon of 0.05)
- Perturbation strength (delta of 0.1)
- Statistical properties of the watermarking scheme
"""

from watermark import (
    ShakespeareNanoGPTModel,
    AESPRF,
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
    delta = 0.1  # Perturbation strength
    z_threshold = 2.0  # Z-score threshold for 95% confidence
    
    # Calculate required covertext length
    calculator = SmoothCovertextCalculator()
    required_length = calculator.get_covertext_length(
        n=n_bits,
        epsilon=epsilon,
        delta=delta
    )
    print(f"\nRequired covertext length for {n_bits} bits with {(1-epsilon)*100}% accuracy: {required_length} tokens")
    
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
    
    # Generate watermarked text of required length
    print("\nGenerating watermarked text of required length...")
    watermarked_text, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,
        covertext_length=required_length  # Use calculated length
    )
    
    print(f"\nGenerated text ({len(watermarked_text)} characters):")
    print(watermarked_text)
    
    # Extract watermark
    print("\nExtracting watermark...")
    recovered_counters, _ = extractor.extract(
        keys=keys,
        h=history,
        ct=watermarked_text,
        c=c
    )
    
    # Detect watermark bits using proper statistical test
    recovered_message = []
    for counter in recovered_counters:
        bit = detect(required_length, counter, len(message), epsilon) # modify so delta is passed in too
        recovered_message.append(1 if bit else 0)  # Convert True/False to 1/0
    
    print("\nResults:")
    print(f"Original message: {message}")
    print(f"Recovered message: {recovered_message}")
    print(f"Success: {message == recovered_message}")

if __name__ == "__main__":
    main() 