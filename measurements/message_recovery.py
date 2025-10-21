import numpy as np
import matplotlib.pyplot as plt
from watermark import (
    GPT2Model,
    AESPRF, 
    SmoothPerturb,
    SmoothCovertextCalculator,
    Embedder,
    Extractor
)
from watermark.utils import detect

def measure_recovery_rates(
    num_trials: int = 100,
    message_sizes: list[int] = [1, 2, 4, 8, 16],
    epsilon: float = 0.05,  # 95% success probability
    delta: float = 0.1,     # Perturbation strength
    safety_factor: int = 10
) -> tuple[list[float], list[float]]:
    """
    Measure message recovery success rates for different message sizes
    """
    # Initialize components
    model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)
    calculator = SmoothCovertextCalculator()

    success_rates = []
    confidence_intervals = []
    
    for msg_size in message_sizes:
        trial_successes = []
        
        # Calculate required covertext length for this message size
        required_length = calculator.get_covertext_length(
            n=msg_size,
            epsilon=epsilon,
            delta=delta,
            safety_factor=safety_factor
        )
        
        for _ in range(num_trials):
            # Generate random message
            message = np.random.randint(0, 2, msg_size).tolist()
            
            # Generate keys (one per bit)
            keys = [bytes([i] * 32) for i in range(msg_size)]
            
            # Setup history
            history = ["This is a test message for watermarking."]
            
            # Generate watermarked text
            watermarked_text, _, _ = embedder.embed(
                keys=keys,
                h=history,
                m=message,
                delta=delta,
                c=5,  # n-gram length
                covertext_length=required_length
            )
            
            # Extract watermark
            recovered_counters, _ = extractor.extract(
                keys=keys,
                h=history,
                ct=watermarked_text,
                c=5
            )
            
            # Detect watermark bits
            recovered_message = []
            for counter in recovered_counters:
                bit = detect(required_length, counter, msg_size, epsilon)
                recovered_message.append(1 if bit else 0)
            
            # Calculate success rate
            success = np.mean([a == b for a, b in zip(message, recovered_message)])
            trial_successes.append(success)
            
        # Calculate statistics
        mean_success = np.mean(trial_successes)
        ci = 1.96 * np.std(trial_successes) / np.sqrt(num_trials)
        
        success_rates.append(mean_success)
        confidence_intervals.append(ci)
        
    return success_rates, confidence_intervals

def plot_recovery_results(
    message_sizes: list[int],
    success_rates: list[float],
    confidence_intervals: list[float],
    output_path: str = "watermarks/measurements/figures/message_recovery.png"
):
    """Plot message recovery results with error bars"""
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        message_sizes,
        success_rates,
        yerr=confidence_intervals,
        fmt='o-',
        capsize=5,
        label='Recovery Rate'
    )
    
    plt.xlabel('Message Size (bits)')
    plt.ylabel('Recovery Success Rate')
    plt.title('Message Recovery Success vs Message Size')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

def main():
    message_sizes = [1, 2, 4]
    success_rates, confidence_intervals = measure_recovery_rates(
        message_sizes=message_sizes
    )
    
    plot_recovery_results(
        message_sizes,
        success_rates,
        confidence_intervals
    )
    
    # Print numerical results
    print("\nMessage Recovery Results:")
    print("Message Size | Success Rate ± 95% CI")
    print("-" * 35)
    for size, rate, ci in zip(message_sizes, success_rates, confidence_intervals):
        print(f"{size:11d} | {rate:.3f} ± {ci:.3f}")

if __name__ == "__main__":
    main() 