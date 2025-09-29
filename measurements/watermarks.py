import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import openai
from watermark import (
    AESPRF,
    Embedder,
    Extractor,
    GPT2Model,
    NGramShuffleAttack,
    ParaphraseAttack,
    SmoothCovertextCalculator,
    SmoothPerturb,
    SynonymAttack,
)


#! Add function to look at chance of repeated n-grams in covertext as n varies
def distinct_n(tokens, n):
    """
    Calculate the distinct-n metric for a sequence of tokens.
    
    Args:
        tokens (list of str): The sequence of tokens.
        n (int): The n-gram length.
    
    Returns:
        float: The distinct-n value, defined as the number of unique n-grams divided by 
               the total number of n-grams. Returns 0.0 if there are fewer tokens than n.
    """
    # Check if there are enough tokens to form at least one n-gram
    if len(tokens) < n:
        return 0.0

    # Generate all n-grams (each n-gram is a tuple of n tokens)
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # Count the unique n-grams using a set
    unique_ngrams = set(ngrams)
    
    # Calculate the distinct-n metric
    distinct_value = len(unique_ngrams) / len(ngrams)
    return distinct_value

def generate_covertext(length: int) -> list[str]:
    """
    Generate unperturbed covertext of specified length using watermark system with all-zero message
    
    Args:
        length (int): Desired length of covertext in tokens
    
    Returns:
        list[str]: List of tokens in generated covertext
    """
    # Initialize components
    model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    
    # Setup for unperturbed generation (all zeros message)
    message = [0] * 3  # Using 3 bits but all zeros
    keys = [bytes([i] * 32) for i in range(3)]
    history = ["This is a test message."]
    delta = 0.1  # Won't matter since message is all zeros
    
    # Generate text
    _, tokens, _ = embedder.embed(
        keys=keys,
        h=history,
        m=message,
        delta=delta,
        c=5,
        covertext_length=length
    )
    
    return tokens['input_ids'][0].tolist()

def plot_repeated_ngrams(
    n_gram_values: list[int] = [3, 4, 5, 6, 7],
    covertext_length: int = 100,
    num_samples: int = 100,
    output_path: str = "./figures/watermarks_repeated_ngrams.png"
):
    """
    For each n-gram value, generate multiple covertexts and measure the repeated n-grams.
    
    Args:
        n_gram_values (list[int]): List of n-gram sizes to test
        covertext_length (int): Length of each covertext to generate
        num_samples (int): Number of covertexts to generate for each n
        output_path (str): Where to save the plot
    """
    # Store results
    distinct_values = []
    confidence_intervals = []
    
    # For each n-gram size
    for i, n in enumerate(n_gram_values):
        print(f"\nProcessing n-gram size {n} ({i+1}/{len(n_gram_values)})")
        sample_values = []
        
        # Generate multiple covertexts
        for j in range(num_samples):
            if j % 10 == 0:  # Print every 10th sample
                print(f"  Sample {j+1}/{num_samples}")
            tokens = generate_covertext(covertext_length)
            distinct_value = distinct_n(tokens, n)
            sample_values.append(distinct_value)
            
        # Calculate mean and confidence interval
        mean_value = np.mean(sample_values)
        ci = 1.96 * np.std(sample_values) / np.sqrt(num_samples)
        
        distinct_values.append(mean_value)
        confidence_intervals.append(ci)
        print(distinct_values)
        print(confidence_intervals)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        n_gram_values,
        distinct_values,
        yerr=confidence_intervals,
        fmt='o-',
        capsize=5
    )
    
    plt.xlabel('n-gram Size')
    plt.ylabel('Distinct-n Ratio')
    plt.title('Uniqueness of n-grams vs n-gram Size')
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    
    # Print numerical results
    print("\nDistinct n-gram Ratios:")
    print("n-gram Size | Ratio ± 95% CI")
    print("-" * 30)
    for n, ratio, ci in zip(n_gram_values, distinct_values, confidence_intervals):
        print(f"{n:11d} | {ratio:.3f} ± {ci:.3f}")


def plot_watermark_length_delta(delta_values, n_bits, epsilon=0.05, safety_factor=1,
                                output_path: str = "./figures/watermarks_length_delta.png"):
    """
    Plots the covertext length required as a function of perturbation strength δ,
    saves the graph as a PNG file, and writes the underlying data to a TXT file.

    Parameters:
      - delta_values (list or np.array): Array of δ values.
      - n_bits (int): Length of the message to hide.
      - epsilon (float): Desired failure probability (default 0.05).
      - safety_factor (float): Multiplier for covertext length (default 1).
      - output_path (str): Full path for the PNG output file.
          Default is "./figures/watermarks_length_delta.png".  
          The corresponding TXT file will be saved at the same location with a .txt extension.
    """
    # Derive the TXT filename by replacing the .png extension with .txt
    txt_filename = output_path.replace(".png", ".txt")
    
    # Create the output directory if it does not exist.
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    
    # Instantiate your covertext length calculator.
    calculator = SmoothCovertextCalculator()
    
    # Calculate covertext lengths for each δ value.
    covertext_lengths = []
    for delta in delta_values:
        required_length = calculator.get_covertext_length(
            n=n_bits,
            epsilon=epsilon,
            delta=delta,
            safety_factor=safety_factor
        )
        covertext_lengths.append(required_length)
    
    # Plot the results.
    plt.figure(figsize=(8, 6))
    plt.plot(delta_values, covertext_lengths, marker='o', linestyle='-')
    plt.xlabel("Perturbation Strength (δ)")
    plt.ylabel("Required Covertext Length (tokens)")
    plt.title(f"Covertext Length vs. δ (n = {n_bits}, ε = {epsilon}, Safety Factor = {safety_factor})")

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Configure the y-axis to display more granular tick labels.
    # ax = plt.gca()
    # Set major ticks at powers of 10.
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    # Set minor ticks for intermediate values (e.g. 2, 3, ..., 9 times each power of 10).
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=10))
    # Use a formatter that shows labels for both major and minor ticks.
    # ax.yaxis.set_minor_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=False))

    plt.grid(True)
    
    # Save the figure to a PNG file.
    plt.savefig(output_path)
    plt.close()
    
    # Save the underlying data to a TXT file.
    data = np.column_stack((delta_values, covertext_lengths))
    header = "δ, Required_Covertext_Length"
    np.savetxt(txt_filename, data, header=header, delimiter=",", fmt="%.6f")
    print(f"Graph saved as: {output_path}\nData saved as: {txt_filename}")


def plot_watermark_length_m(m_lengths, delta, epsilon=0.05, safety_factor=1,
                            output_path: str = "./figures/watermarks_length_m.png"):
    """
    Plots the covertext length required as a function of message length,
    saves the graph as a PNG file, and writes the underlying data to a TXT file.

    Parameters:
      - m_lengths (list or np.array): Array of message lengths.
      - delta (float): Fixed perturbation strength.
      - epsilon (float): Desired failure probability (default 0.05).
      - safety_factor (float): Multiplier for covertext length (default 1).
      - output_path (str): Full path for the PNG output file.
          Default is "./figures/watermarks_length_m.png".  
          The corresponding TXT file will be saved at the same location with a .txt extension.
    """
    # Derive the TXT filename by replacing the .png extension with .txt
    txt_filename = output_path.replace(".png", ".txt")
    
    # Create the output directory if it does not exist.
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    
    # Instantiate your covertext length calculator.
    calculator = SmoothCovertextCalculator()
    
    # Calculate covertext lengths for each message length.
    covertext_lengths = []
    for m in m_lengths:
        required_length = calculator.get_covertext_length(
            n=m,
            epsilon=epsilon,
            delta=delta,
            safety_factor=safety_factor
        )
        covertext_lengths.append(required_length)
    
    # Plot the results.
    plt.figure(figsize=(8, 6))
    plt.plot(m_lengths, covertext_lengths, marker='o', linestyle='-')
    plt.xlabel("Message Length (n bits)")
    plt.ylabel("Required Covertext Length (tokens)")
    plt.title(f"Covertext Length vs. Message Length (δ = {delta}, ε = {epsilon}, Safety Factor = {safety_factor})")
    plt.grid(True)
    
    # Save the figure to a PNG file.
    plt.savefig(output_path)
    plt.close()
    
    # Save the underlying data to a TXT file.
    data = np.column_stack((m_lengths, covertext_lengths))
    header = "Message_Length, Required_Covertext_Length"
    np.savetxt(txt_filename, data, header=header, delimiter=",", fmt="%.6f")
    print(f"Graph saved as: {output_path}\nData saved as: {txt_filename}")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_recovery_accuracy(original, recovered):
    """
    Computes bitwise accuracy between two lists of bits.
    Returns the fraction of matching bits.
    """
    if len(original) != len(recovered):
        return 0.0
    correct = sum(1 for o, r in zip(original, recovered) if o == r)
    return correct / len(original)


def apply_partial_paraphrase(text, paraphrase_attack, tampering_percentage):
    """
    Applies the paraphrase attack only on a fraction of sentences.
    
    Splits the text into sentences (using punctuation as delimiters), then for
    each sentence, with probability equal to tampering_percentage the paraphrase
    attack is applied; otherwise, the sentence is left unchanged.
    Returns the recombined text.
    """
    # Simple splitting on sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    new_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if random.random() < tampering_percentage:
            new_sentence = paraphrase_attack(sentence)
            new_sentences.append(new_sentence)
        else:
            new_sentences.append(sentence)
    return " ".join(new_sentences)


# -----------------------------------------------------------------------------
# Main experimental function
# -----------------------------------------------------------------------------
def plot_watermarking_tampering_effect(tampering_types, tampering_percentages, baseline_recovery,
                                       output_path: str = "./figures/watermarks_"):
    """
    Runs the watermarking system, attacks the watermarked text with the specified types and tampering percentages,
    and measures two metrics for each condition:
    
      1. Average bitwise recovery accuracy (from one run).
      2. Perfect recovery rate (percentage of 100 runs where the entire message is recovered perfectly).
      
    For each attack type (and for each mode if applicable) two graphs are generated:
      - One showing bitwise accuracy versus tampering percentage (with a horizontal dashed line at the baseline recovery).
      - One showing perfect recovery rate versus tampering percentage (with a horizontal dashed line at 100% perfect recovery).
      
    Both graphs and their corresponding data files (TXT) are saved in ./figures/ with descriptive filenames.
    
    Parameters:
      - tampering_types (list): List of attack types to evaluate (e.g., ["NGram Shuffle", "Synonym Attack", "Paraphrase Attack"]).
      - tampering_percentages (list): List of tampering percentages (floats between 0 and 1).
      - baseline_recovery (float): Baseline recovery accuracy (no attack) for bitwise measurement.
      - output_path (str): Base filename prefix for saving outputs.
    """
    # -----------------------------------------------------------------------------
    # Setup watermarking system (using your provided parameters)
    # -----------------------------------------------------------------------------
    n_bits = 1              # Length of message to hide
    epsilon = 1 - baseline_recovery          # 95% success probability
    delta = 0.2             # Perturbation strength
    safety_factor = 10
    c = 5                   # n-gram length for PRF used by watermarking
    
    # Instantiate required components.
    calculator = SmoothCovertextCalculator()
    required_length = calculator.get_covertext_length(n=n_bits, epsilon=epsilon, delta=delta, safety_factor=safety_factor)
    print(f"\nRequired covertext length for {n_bits} bits with {(1-epsilon)*100}% accuracy: {required_length} tokens")
    
    model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size - 1)
    perturb = SmoothPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)
    
    # Watermarking parameters.
    message = [1, 0, 1]  # 3-bit message to hide
    keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
    history = ["How have you been?"]  # Context
    
    # Generate watermarked text.
    print("\nGenerating watermarked text of required length...")
    watermarked_text, _, _ = embedder.embed(keys=keys, h=history, m=message, delta=delta, c=c, covertext_length=required_length)
    
    # -----------------------------------------------------------------------------
    # For each attack type, run experiments.
    # -----------------------------------------------------------------------------
    for attack_type in tampering_types:
        # Decide which modes to test.
        # For NGram Shuffle and Paraphrase Attack, test both local (True) and global (False).
        # For Synonym Attack, only test global (local=False).
        if attack_type == "NGram Shuffle" or attack_type == "Paraphrase Attack":
            mode_list = [True, False]   # True: local; False: global.
        elif attack_type == "Synonym Attack":
            mode_list = [False]
        else:
            print(f"Unknown attack type: {attack_type}. Skipping.")
            continue
        
        # Initialize dictionaries to store metrics (keys will be "local" or "global")
        results_bitwise = {("local" if mode else "global"): [] for mode in mode_list}
        results_perfect = {("local" if mode else "global"): [] for mode in mode_list}
        data_lines_bitwise = ["Tampering_Percentage\tMode\tBitwise_Accuracy"]
        data_lines_perfect = ["Tampering_Percentage\tMode\tPerfect_Recovery_Rate"]
        
        # For each tampering percentage...
        for tp in tampering_percentages:
            # For each mode (local/global)
            for mode in mode_list:
                mode_label = "local" if mode else "global"
                # --- Bitwise measurement (single run) ---
                if attack_type == "NGram Shuffle":
                    attack = NGramShuffleAttack(model=model, n=3, probability=tp, local=mode)
                    attacked_text = attack(watermarked_text)
                elif attack_type == "Synonym Attack":
                    attack = SynonymAttack(method="wordnet", probability=tp)
                    attacked_text = attack(watermarked_text)
                elif attack_type == "Paraphrase Attack":
                    # Create the paraphrase attack instance.
                    client = openai.OpenAI()
                    attack_instance = ParaphraseAttack(client=client, model="gpt-4o-mini", temperature=0.0, local=mode)
                    # For partial attack, apply the paraphrase only on a fraction of sentences.
                    if tp < 1.0:
                        attacked_text = apply_partial_paraphrase(watermarked_text, attack_instance, tp)
                    else:
                        attacked_text = attack_instance(watermarked_text)
                else:
                    attacked_text = watermarked_text  # Fallback; should not occur.
                
                recovered_message = extractor.extract(keys=keys, ct=attacked_text, h=history, c=c)
                bitwise_acc = compute_recovery_accuracy(message, recovered_message)
                results_bitwise[mode_label].append(bitwise_acc)
                data_lines_bitwise.append(f"{tp}\t{mode_label}\t{bitwise_acc}")
                
                # --- Perfect recovery measurement (100 runs) ---
                perfect_count = 0
                runs = 100
                for _ in range(runs):
                    if attack_type == "NGram Shuffle":
                        attack = NGramShuffleAttack(model=model, n=3, probability=tp, local=mode)
                        attacked_text = attack(watermarked_text)
                    elif attack_type == "Synonym Attack":
                        attack = SynonymAttack(method="wordnet", probability=tp)
                        attacked_text = attack(watermarked_text)
                    elif attack_type == "Paraphrase Attack":
                        client = openai.OpenAI()
                        attack_instance = ParaphraseAttack(client=client, model="gpt-4o-mini", temperature=0.0, local=mode)
                        if tp < 1.0:
                            attacked_text = apply_partial_paraphrase(watermarked_text, attack_instance, tp)
                        else:
                            attacked_text = attack_instance(watermarked_text)
                    else:
                        attacked_text = watermarked_text
                    
                    recovered_message = extractor.extract(attacked_text)
                    if recovered_message == message:  # Perfect recovery if the lists match exactly.
                        perfect_count += 1
                perfect_rate = perfect_count / runs
                results_perfect[mode_label].append(perfect_rate)
                data_lines_perfect.append(f"{tp}\t{mode_label}\t{perfect_rate}")
        
        # -----------------------------------------------------------------------------
        # Plot Bitwise Recovery Accuracy for this attack type.
        # -----------------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.axhline(baseline_recovery, color='gray', linestyle='--', label="Baseline")
        markers = {'local': 'o', 'global': 's'}
        linestyles = {'local': '-', 'global': '--'}
        for mode_label, acc_list in results_bitwise.items():
            plt.plot(tampering_percentages, acc_list, marker=markers.get(mode_label, 'o'),
                     linestyle=linestyles.get(mode_label, '-'), label=mode_label)
        plt.xlabel("Tampering Percentage")
        plt.ylabel("Bitwise Recovery Accuracy")
        plt.title(f"{attack_type} Attack (Bitwise Accuracy)")
        plt.legend()
        plt.grid(True)
        attack_fname = attack_type.lower().replace(" ", "_")
        png_filename_bitwise = f"{output_path}{attack_fname}_bitwise.png"
        txt_filename_bitwise = f"{output_path}{attack_fname}_bitwise.txt"
        plt.savefig(png_filename_bitwise)
        plt.close()
        print(f"Bitwise graph saved as: {png_filename_bitwise}")
        with open(txt_filename_bitwise, 'w') as f:
            f.write("\n".join(data_lines_bitwise))
        print(f"Bitwise data saved as: {txt_filename_bitwise}")
        
        # -----------------------------------------------------------------------------
        # Plot Perfect Recovery Rate for this attack type.
        # -----------------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.axhline(1.0, color='gray', linestyle='--', label="100% Perfect Recovery")
        for mode_label, rate_list in results_perfect.items():
            plt.plot(tampering_percentages, rate_list, marker=markers.get(mode_label, 'o'),
                     linestyle=linestyles.get(mode_label, '-'), label=mode_label)
        plt.xlabel("Tampering Percentage")
        plt.ylabel("Perfect Recovery Rate")
        plt.title(f"{attack_type} Attack (Perfect Recovery Rate)")
        plt.legend()
        plt.grid(True)
        png_filename_perfect = f"{output_path}{attack_fname}_perfect.png"
        txt_filename_perfect = f"{output_path}{attack_fname}_perfect.txt"
        plt.savefig(png_filename_perfect)
        plt.close()
        print(f"Perfect recovery graph saved as: {png_filename_perfect}")
        with open(txt_filename_perfect, 'w') as f:
            f.write("\n".join(data_lines_perfect))
        print(f"Perfect recovery data saved as: {txt_filename_perfect}")

def main():
    # plot_repeated_ngrams()

    # ---------------------------
    # Use message lengths [1, 2, 3, 4, 8, 16] with a fixed delta value of 0.1.
    # ---------------------------
    # m_lengths = [1, 2, 3, 4, 8, 16]
    # fixed_delta = 0.1  # The fixed perturbation strength for this experiment
    # Call the function; default epsilon and safety_factor are used unless specified otherwise.
    # plot_watermark_length_m(m_lengths, delta=fixed_delta)
    
    # ---------------------------
    # Plot 2: Covertext Length vs. Delta
    # Use delta values [0.01, 0.05, 0.1, 0.2] with a fixed message length of n = 3.
    # ---------------------------
    # delta_values = [0.01, 0.05, 0.1, 0.2]
    # n_bits = 1  # Fixed message length for this experiment
    # plot_watermark_length_delta(delta_values, n_bits)

    # ---------------------------
    tampering_types = ["NGram Shuffle", "Synonym Attack", "Paraphrase Attack"]
    tampering_percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    recovery_accuracy = 0.95
    plot_watermarking_tampering_effect(tampering_types, tampering_percentages, recovery_accuracy)

if __name__ == "__main__":
    main()

