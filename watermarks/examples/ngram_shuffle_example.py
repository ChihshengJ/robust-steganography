"""
Example demonstrating how the n-gram shuffle attack affects watermark recovery.
"""

from watermark import (
    ShakespeareNanoGPTModel,
    HMACPRF,
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)
from watermark.attacks.ngram_shuffle import NGramShuffleAttack
from watermark.utils import detect

def main():
    # Initialize components
    model = ShakespeareNanoGPTModel()
    calculator = SmoothCovertextCalculator()
    perturb = SmoothPerturb()
    prf = HMACPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Setup watermarking parameters
    message = [1, 0, 1]  # 3-bit message
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]
    history = ["To be, or not to be- that is the question:"]
    delta = 0.2  # Perturbation strength
    epsilon = 0.05  # 95% success probability
    safety_factor = 10
    c = 5
    
    # Calculate required covertext length
    required_length = calculator.get_covertext_length(
        n=len(message),
        epsilon=epsilon,
        delta=delta,
        safety_factor=safety_factor
    )
    print(f"Required covertext length: {required_length} tokens")
    
    # Generate watermarked text
    watermarked_text, _, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,
        covertext_length=required_length
    )
    
    # print("\nOriginal watermarked text:")
    # print(watermarked_text)

    # Create and apply attack
    attack = NGramShuffleAttack(model=model, n=c+1, probability=0.3, local=False)
    attacked_text = attack(watermarked_text)
    
    # print("\nText after n-gram shuffle attack:")
    # print(attacked_text)

    # Try to recover the message
    recovered_counters, _ = extractor.extract(
        keys=keys,
        h=history,
        ct=attacked_text,
        c=5
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