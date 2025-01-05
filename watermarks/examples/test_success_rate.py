"""
Test script to measure watermark recovery success rate over multiple runs.
Runs the watermarking system repeatedly and reports the achieved success rate.
"""
import os

from watermark import (
    ShakespeareNanoGPTModel,
    HMACPRF,
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)
from watermark.utils import detect
from tqdm import tqdm

def run_single_test(
    embedder,
    extractor,
    message,
    keys,
    history,
    required_length,
    delta,
    c,
    epsilon
) -> bool:
    """Run a single test and return True if message was recovered correctly"""
    
    # Generate watermarked text
    watermarked_text, _, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=c,
        covertext_length=required_length
    )
    
    # Extract watermark
    recovered_counters, _ = extractor.extract(
        keys=keys,
        h=history,
        ct=watermarked_text,
        c=c
    )
    
    # Detect watermark bits
    recovered_message = []
    for counter in recovered_counters:
        bit = detect(required_length, counter, len(message), epsilon)
        recovered_message.append(1 if bit else 0)
    
    return message == recovered_message

def main():
    # Test parameters
    DESIRED_SUCCESS_RATE = 0.95  # 95% success rate target
    N_TRIALS = 100  # Number of tests to run
    
    # Watermarking parameters
    n_bits = 3  # Length of message to hide
    epsilon = 0.05  # For 95% success probability
    delta = 0.2  # Perturbation strength
    safety_factor = 10
    message = [1, 0, 1]  # Test message
    keys = [os.urandom(32) for _ in range(3)]  # Random 32-byte keys, one per bit
    history = ["To be, or not to be- that is the question:"]
    c = 5  # Length of n-grams for PRF
    
    # Initialize components
    print("Initializing components...")
    model = ShakespeareNanoGPTModel()
    prf = HMACPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)
    
    # Calculate required text length
    calculator = SmoothCovertextCalculator()
    required_length = calculator.get_covertext_length(
        n=n_bits,
        epsilon=epsilon,
        delta=delta,
        safety_factor=safety_factor
    )
    print(f"Required length: {required_length} tokens")
    
    # Run trials
    print(f"\nRunning {N_TRIALS} trials...")
    successes = 0
    
    for _ in tqdm(range(N_TRIALS)):
        success = run_single_test(
            embedder=embedder,
            extractor=extractor,
            message=message,
            keys=keys,
            history=history,
            required_length=required_length,
            delta=delta,
            c=c,
            epsilon=epsilon
        )
        if success:
            successes += 1
    
    # Calculate and report results
    achieved_rate = successes / N_TRIALS
    passed = achieved_rate >= DESIRED_SUCCESS_RATE
    
    print(f"\nResults:")
    print(f"Desired success rate: {DESIRED_SUCCESS_RATE:.1%}")
    print(f"Achieved success rate: {achieved_rate:.1%}")
    print(f"Passed: {passed}")

if __name__ == "__main__":
    main() 