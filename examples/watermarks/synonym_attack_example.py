"""
Example demonstrating how the synonym attack affects watermark recovery.
"""

from watermark import (
    ShakespeareNanoGPTModel,
    HMACPRF,
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)
from watermark.attacks.synonym import SynonymAttack
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
    # required_length = 100
    
    # Initialize components
    model = ShakespeareNanoGPTModel()
    prf = HMACPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)
    attack = SynonymAttack(method="wordnet", probability=0.2)

    # Setup watermarking parameters
    message = [1, 0, 1]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["To be, or not to be- that is the question:"]  # Shakespeare-style context
    c = 5  # Length of n-grams used by PRF for watermarking
    
    print("\nGenerating watermarked text...")
    watermarked_text, _, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,
        covertext_length=required_length
    )
    
    print(f"\nWatermarked text ({len(watermarked_text)} characters):")
    print(watermarked_text)
    
    # Apply synonym attack
    attacked_text = attack(watermarked_text)
    print("\nText after synonym attack:")
    print(attacked_text)
    
    # Extract watermark from attacked text
    print("\nExtracting watermark from attacked text...")
    recovered_counters, _ = extractor.extract(
        keys=keys,
        h=history,
        ct=attacked_text,
        c=c
    )
    
    # Detect watermark bits
    recovered_message = []
    for counter in recovered_counters:
        bit = detect(required_length, counter, len(message), epsilon)
        recovered_message.append(1 if bit else 0)
    
    print("\nResults:")
    print(f"Original message: {message}")
    print(f"Recovered message: {recovered_message}")
    print(f"Success: {message == recovered_message}")

if __name__ == "__main__":
    main() 