"""
Measure how n-gram length and perturbation strength affect the watermarking system
"""

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

def measure_covertext_lengths(
    message_size: int = 3,
    epsilon: float = 0.05,  # 95% success probability
    deltas: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    ngram_sizes: list[int] = [3, 4, 5, 6, 7],
    safety_factor: int = 10
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Measure required covertext lengths for different n-gram sizes and deltas
    """
    calculator = SmoothCovertextCalculator()
    
    # Matrix of required lengths
    lengths = []
    
    for c in ngram_sizes:
        c_lengths = []
        for delta in deltas:
            length = calculator.get_covertext_length(
                n=message_size,
                epsilon=epsilon,
                delta=delta,
                safety_factor=safety_factor
            )
            c_lengths.append(length)
        lengths.append(c_lengths)
            
    return lengths

def plot_parameter_effects(
    deltas: list[float],
    ngram_sizes: list[int],
    lengths: list[list[int]],
    output_path: str = "watermarks/measurements/figures/parameter_effects.png"
):
    """Plot how parameters affect required covertext length"""
    
    plt.figure(figsize=(12, 6))
    
    for i, c in enumerate(ngram_sizes):
        plt.plot(deltas, lengths[i], 'o-', label=f'n-gram size = {c}')
    
    plt.xlabel('Perturbation Strength (δ)')
    plt.ylabel('Required Covertext Length (tokens)')
    plt.title('Required Length vs Perturbation Strength\nfor Different n-gram Sizes')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

def main():
    deltas = [0.05, 0.1, 0.15, 0.2, 0.25]
    ngram_sizes = [3, 4, 5, 6, 7]
    
    lengths = measure_covertext_lengths(
        deltas=deltas,
        ngram_sizes=ngram_sizes
    )
    
    plot_parameter_effects(
        deltas,
        ngram_sizes,
        lengths
    )
    
    # Print numerical results
    print("\nRequired Covertext Lengths:")
    print("Delta | " + " | ".join(f"n-gram={c}" for c in ngram_sizes))
    print("-" * (7 + 12 * len(ngram_sizes)))
    for i, delta in enumerate(deltas):
        row = [f"{delta:.2f}"]
        for j in range(len(ngram_sizes)):
            row.append(f"{lengths[j][i]:8d}")
        print(" | ".join(row))

if __name__ == "__main__":
    main() 