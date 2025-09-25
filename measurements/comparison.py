# comparison_graphs.py
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from watermark import SmoothCovertextCalculator

def plot_token_efficiency_comparison():
    # Ensure the figures directory exists.
    # os.makedirs("./figures", exist_ok=True)
    
    # --------- Parameters ---------
    # Watermarking parameters
    watermarking_deltas = [0.05, 0.1, 0.2]  # List of delta values to try
    epsilon = 0.05       # Fixed probability parameter (95% success)
    safety_factor = 1    # Fixed safety factor

    # Embedding parameters
    embedding_hash_lengths = [1, 4]           # Number of bits hidden per chunk
    embedding_covertext_chunk_lengths = [100, 200]      # Different covertext chunk lengths (in tokens)

    # Range of hidden message lengths (in bits) to test.
    message_lengths = range(1, 51)  # For example, from 1 to 50 bits

    # --------- Data Calculation ---------
    # Watermarking data: for each delta, compute required covertext length for each message length.
    calculator = SmoothCovertextCalculator()
    watermarking_data = {}
    for delta in watermarking_deltas:
        watermarking_data[delta] = {"message_lengths": [], "required_tokens": []}
        for n in message_lengths:
            required_length = calculator.get_covertext_length(
                n=n, epsilon=epsilon, delta=delta, safety_factor=safety_factor
            )
            watermarking_data[delta]["message_lengths"].append(n)
            watermarking_data[delta]["required_tokens"].append(required_length)
    
    # Embedding data: for each combination of (hash_length, covertext_chunk_length)
    # the tokens required is: covertext_chunk_length * ceil(message_length / hash_length)
    embedding_data = {}
    for chunk_length in embedding_covertext_chunk_lengths:
        for hash_length in embedding_hash_lengths:
            key = (hash_length, chunk_length)
            embedding_data[key] = {"message_lengths": [], "required_tokens": []}
            for n in message_lengths:
                required_tokens = chunk_length * math.ceil(n / hash_length)
                embedding_data[key]["message_lengths"].append(n)
                embedding_data[key]["required_tokens"].append(required_tokens)
    
    # --------- Save Data to File ---------
    data_file = "./figures/token_efficiency_comparison.txt"
    with open(data_file, "w") as f:
        f.write("scheme\tparameter\tmessage_length\trequired_tokens\n")
        # Write watermarking results.
        for delta, data in watermarking_data.items():
            for n, tokens in zip(data["message_lengths"], data["required_tokens"]):
                f.write(f"watermarking\tdelta={delta}\t{n}\t{tokens}\n")
        # Write embedding results.
        for (hash_length, chunk_length), data in embedding_data.items():
            for n, tokens in zip(data["message_lengths"], data["required_tokens"]):
                f.write(f"embedding\thash_length={hash_length},chunk_length={chunk_length}\t{n}\t{tokens}\n")
    
    # --------- Plotting ---------
    plt.figure(figsize=(10, 6))
    
    # Define some markers to differentiate lines.
    markers = ['o', 's', 'v', 'D', '^', 'x']
    
    # Plot watermarking data with solid lines.
    for i, (delta, data) in enumerate(watermarking_data.items()):
        plt.plot(
            data["message_lengths"], data["required_tokens"],
            marker=markers[i % len(markers)], linestyle='-',
            label=f"Watermarking (delta={delta})"
        )
    
    # Plot embedding data with dashed lines.
    for i, ((hash_length, chunk_length), data) in enumerate(embedding_data.items()):
        plt.plot(
            data["message_lengths"], data["required_tokens"],
            marker=markers[i % len(markers)], linestyle='--',
            label=f"Embedding (hash_length={hash_length}, chunk_length={chunk_length})"
        )
    
    plt.xlabel("Number of hidden bits")
    plt.ylabel("Number of required tokens")
    plt.title("Token Efficiency Comparison")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot.
    plot_file = "./figures/token_efficiency_comparison.png"
    plt.savefig(plot_file)
    plt.close()

def plot_tampering_effect_comparison(tampering_levels, recovery_watermarking, recovery_embedding_1bit):
    """
    Plots message recovery accuracy for both systems under adversarial tampering.

    Parameters:
    - tampering_levels (list or np.array): Different levels of tampering (e.g., edits, synonym swaps).
    - recovery_watermarking (list or np.array): Percentage of successful recovery in watermarking system.
    - recovery_embedding_1bit (list or np.array): Percentage of successful recovery in 1-bit LSH embedding system.

    Goal:
    - Show which system is more robust under different types of tampering.
    
    # TODO: Implement function to generate line plot.
    """
    pass

if __name__ == "__main__":
    plot_token_efficiency_comparison()
    # plot_tampering_effect_comparison()
