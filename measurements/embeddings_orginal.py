# embedding_graphs.py

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import openai
from tqdm import tqdm

from embeddings import (
    CORPORATE_MONOLOGUE,
    MinimalEncoder,
    RepetitionCode,
    StegSystem,
)
from embeddings.utils.get_embedding import (
    get_embeddings_in_batch,
)
from embeddings.utils.new_text import generate_response
from watermarks import (
    GPT2Model,
    LanguageModel,
    NGramShuffleAttack,
    ParaphraseAttack,
    SynonymAttack,
)

from .watermarks import apply_partial_paraphrase, compute_recovery_accuracy

#! change to:
#! 1. plot_rejection_sampling_efficiency - orange vs blue for 1 bit and then purple for simulation n > 1
#! 2. plot_embedding_drift - pick a sequence of 10 samples and plot the drift
#! 3. plot lsh_accuracy - 1bit random vs pca

#! change name of rejection sampling function and split data generation + saving from plotting

#! maybe change so that 1 vs n is not seperated, but n > 1 is labeled as simulation (words, caption, different color, etc.)


def sample(
    client,
    history,
    system_prompt="You are having a casual conversation.",
    max_length=200,
):
    # Generate single response
    message = generate_response(client, history, system_prompt, max_length)

    # Get embedding
    embedding = get_embeddings_in_batch(client, [message])[0]

    # Process embedding
    emb = np.array(embedding).reshape(1, -1)

    return emb


def plot_rejection_sampling_efficiency_1bit(lsh_list):
    """
    Plots rejection sampling efficiency for the real 1-bit LSH embedding system.

    Parameters:
    - trials (list or np.array): Number of independent trials.
    - iterations_needed (list or np.array): Number of iterations needed to find a valid stegotext.

    Goal:
    - Show the efficiency of rejection sampling for encoding hidden bits.

    # TODO: Implement function to generate histogram.
    """
    client = openai.OpenAI()
    history = [
        "I wanted to follow up regarding the implementation timeline for the new risk management system. Based on our initial assessment, we'll need to coordinate closely with both IT and Operations to ensure a smooth transition. Please review the attached documentation when you have a moment.",
        "After consulting with the development team, we've identified several key milestones that need to be addressed before proceeding. The current testing phase has revealed some potential integration issues with our legacy systems, particularly in the trade validation module. We're working on implementing the necessary fixes and expect to have an updated timeline by end of week.",
        "Given the complexity of these changes, I believe it would be beneficial to schedule a stakeholder review meeting. We should include representatives from Risk Management, IT Operations, and the Trading desk to ensure all requirements are being met. I've asked Sarah to coordinate calendars for next Tuesday afternoon.",
    ]
    k = 1

    results = [{"mean": None, "ci": None} for _ in range(8)]
    first_bit_result = {
        **{lsh.__class__.__name__: {"mean": None, "ci": None} for lsh in lsh_list},
    }

    # 1-bit experiments
    for lsh in tqdm(lsh_list, desc="1-bit LSH"):
        hash_fn = lsh
        hash_trials = []
        for _ in tqdm(
            range(50), desc=f"Trials ({lsh.__class__.__name__})", leave=False
        ):
            desired_bits = [random.randint(0, 1)]
            sampled_bits = None
            count = 0
            while not np.array_equal(sampled_bits, desired_bits):
                embedding = sample(client, history, system_prompt=CORPORATE_MONOLOGUE)
                sampled_bits = hash_fn(embedding)
                count += 1
            hash_trials.append(count)

        mean = np.mean(hash_trials)
        ci = 1.96 * np.std(hash_trials) / np.sqrt(len(hash_trials))
        first_bit_result[hash_fn.__class__.__name__] = {"mean": mean, "ci": ci}

    results[0] = first_bit_result

    # n > 1 bits with Oracle
    for n in tqdm(range(2, 9), desc="Oracle n-bit"):
        hash_fn = OracleHash(output_length=n)
        hash_trials = []
        for _ in tqdm(range(50), desc=f"Trials (n={n})", leave=False):
            desired_bits = [random.randint(0, 1) for _ in range(n)]
            sampled_bits = None
            count = 0
            while not np.array_equal(sampled_bits, desired_bits):
                embedding = sample(client, history, system_prompt=CORPORATE_MONOLOGUE)
                sampled_bits = hash_fn(embedding)
                count += 1
            hash_trials.append(count)

        mean = np.mean(hash_trials)
        ci = 1.96 * np.std(hash_trials) / np.sqrt(len(hash_trials))
        results[n - 1] = {"mean": mean, "ci": ci}
    print("Results:", results)

    # Save results to file
    with open("./figures/embeddings_rejection_sampling.txt", "w") as f:
        f.write("Rejection Sampling Results:\n")
        for lsh_name, result in first_bit_result.items():
            f.write(
                f"{lsh_name}: {result['mean']:.2f} ± {result['ci']:.2f} average trials\n"
            )
        for n, result in enumerate(results[1:], start=2):
            f.write(
                f"Oracle {n} bits: {result['mean']:.2f} ± {result['ci']:.2f} average trials\n"
            )
    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot n=1 results (group of 3 bars)
    x_positions = [0, 0.8, 1.6]  # Positions for first group
    labels = list(first_bit_result.keys())

    # Define colors - one for PCA, one for Random, one for all Oracle results
    pca_color = "#1f77b4"  # Blue
    random_color = "#ff7f0e"  # Orange
    oracle_color = "#2ca02c"  # Green

    # Map labels to their colors
    color_map = {
        "PCAHash": pca_color,
        "RandomProjectionHash": random_color,
        "OracleHash": oracle_color,
    }

    # Plot first group (n=1)
    for i, (label, pos) in enumerate(zip(labels, x_positions)):
        result = first_bit_result[label]
        plt.bar(
            pos,
            result["mean"],
            yerr=result["ci"],
            color=color_map[label],
            width=0.6,
            label=label,
        )

    # Plot n > 1 results (single bars)
    for n in range(2, 9):
        result = results[n - 1]
        plt.bar(
            n + 1,
            result["mean"],
            yerr=result["ci"],
            color=oracle_color,
            width=0.6,
            label=f"Oracle {n}-bit",
        )

    # Customize plot
    plt.xlabel("Number of Bits")
    plt.ylabel("Average Number of Trials")
    plt.title("Rejection Sampling Performance")

    # Set x-ticks
    all_positions = x_positions + list(range(3, 10))
    all_labels = ["PCA", "Random", "Oracle"] + [str(i) for i in range(2, 9)]
    plt.xticks(all_positions, all_labels)

    # Add legend
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig("./figures/embeddings_rejection_sampling.png")
    plt.close()


def compute_embedding(message, model: LanguageModel):
    input_tokens = model.tokenizer(message, return_tensors="pt")
    return model.get_next_token_distribution(input_tokens)


def generate_paraphrase_recovery_accuracy(
    tampering_levels,
    system,
    message="ab",
    history=None,
    runs=5,
):
    """
    Generate recovery accuracy (as percentages) for paraphrasing attacks (1-bit LSH).

    Parameters
    ----------
    tampering_levels : list[float]
        Tampering levels between 0.0 and 1.0. If level == 1.0, apply full paraphrase.
    system : StegSystem
        Instance of your StegSystem already configured (hash_fn, ecc, encoder, client, etc).
    message : str
        The short message to hide & recover (defaults to "ab").
    history : list[str] or None
        Context/history to pass to system.hide_message(). If None, uses a small default.
    runs : int
        Number of hide/recover trials per tampering level.

    Returns
    -------
    recovery_accuracy : list[float]
        Per tampering level percentage of successful recoveries (0-100).
    """
    if history is None:
        history = [
            "What are you up to today?",
            "Nothing much, just working on a project.",
            "Want to grab coffee and discuss it?",
        ]

    recovery_accuracy = []
    attack_instance = ParaphraseAttack(
        client=system.client,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    for tp in tqdm(tampering_levels, desc="Tampering levels"):
        success_count = 0
        for _ in tqdm(range(runs), desc=f"runs@tp={tp}", leave=False):
            try:
                stego_texts = system.hide_message(message, history)
            except Exception as e:
                print(f"[Warning] system.hide_message failed: {e}")
                continue

            attacked_stego_texts = []
            for i, stego_text in enumerate(stego_texts):
                if tp < 1.0:
                    attacked_text = apply_partial_paraphrase(
                        stego_text, attack_instance, tp
                    )
                else:
                    attacked_text = attack_instance(message)

                attacked_stego_texts.append(attacked_text)

            try:
                recovered = system.recover_message(attacked_stego_texts)
            except Exception:
                recovered = None

            if recovered == message:
                success_count += 1

        percent = 100.0 * success_count / float(runs) if runs > 0 else 0.0
        recovery_accuracy.append(percent)

    return recovery_accuracy


def plot_embedding_drift(messages, tampering_percentages, runs=1):
    """
    For a list of messages, applies five types of attacks to each message and computes:
      - Euclidean distance between the original embedding and the attacked one.
      - Cosine similarity between the original embedding and the attacked one.

    The five attacks (in order) are:
      1. NGram Shuffle (local)
      2. NGram Shuffle (global)
      3. Synonym Attack (global)
      4. Paraphrase Attack (local)
      5. Paraphrase Attack (global)

    The results are aggregated over all messages and runs.

    Data are saved as:
      "./figures/embedding_drift_euclidean.txt" and
      "./figures/embedding_drift_cosine_similarity.txt"

    Plots are saved as:
      "./figures/embedding_drift_euclidean.png" and
      "./figures/embedding_drift_cosine_similarity.png"

    Parameters:
      - messages (list of str): A list of messages on which to run the experiments.
      - tampering_percentages (list of float): List of tampering percentages (floats between 0 and 1).
      - runs (int): Number of attack runs per message per tampering percentage.
    """
    # -----------------------------------------------------------------------------
    # Setup the model (and any necessary dependencies).
    # -----------------------------------------------------------------------------
    model = GPT2Model()

    # -----------------------------------------------------------------------------
    # Define attack configurations in the required order.
    # Each configuration is a tuple: (attack_label, attack_type, mode)
    # For the synonym attack, mode is not applicable (set to None).
    # -----------------------------------------------------------------------------
    attack_configurations = [
        ("NGram Shuffle (local)", "n-gram", True),
        ("NGram Shuffle (global)", "n-gram", False),
        ("Synonym Attack", "synonym", None),
        ("Paraphrase Attack (local)", "paraphrase", True),
        ("Paraphrase Attack (global)", "paraphrase", False),
    ]

    # -----------------------------------------------------------------------------
    # Initialize dictionaries to store raw metric values.
    # For each attack configuration and each tampering percentage, we store a list of computed values.
    # -----------------------------------------------------------------------------
    results_euclidean = {
        label: {tp: [] for tp in tampering_percentages}
        for label, _, _ in attack_configurations
    }
    results_cosine = {
        label: {tp: [] for tp in tampering_percentages}
        for label, _, _ in attack_configurations
    }

    # Precompute the "original" embeddings for each message.
    original_embeddings = [
        compute_embedding(message, model).detach().numpy() for message in messages
    ]

    # -----------------------------------------------------------------------------
    # Run experiments: For each tampering percentage and each attack configuration,
    # and for each message, run the attack 'runs' times and record the Euclidean distance
    # and cosine similarity between the original and attacked embeddings.
    # -----------------------------------------------------------------------------
    for tp in tampering_percentages:
        print(f"Processing tampering percentage: {tp}")
        for attack_label, attack_type, mode in attack_configurations:
            for i, message in enumerate(messages):
                orig_embed = original_embeddings[i]
                for _ in range(runs):
                    # --- Apply the appropriate attack ---
                    if attack_type == "n-gram":
                        attack = NGramShuffleAttack(
                            model=model, n=3, probability=tp, local=mode
                        )
                        attacked_text = attack(message)
                    elif attack_type == "synonym":
                        attack = SynonymAttack(method="wordnet", probability=tp)
                        attacked_text = attack(message)
                    elif attack_type == "paraphrase":
                        client = openai.OpenAI()
                        attack_instance = ParaphraseAttack(
                            client=client,
                            model="gpt-4o-mini",
                            temperature=0.0,
                            local=mode,
                        )
                        if tp < 1.0:
                            attacked_text = apply_partial_paraphrase(
                                message, attack_instance, tp
                            )
                        else:
                            attacked_text = attack_instance(message)
                    else:
                        attacked_text = message  # Fallback; should not occur.

                    attacked_embed = (
                        compute_embedding(attacked_text, model).detach().numpy()
                    )

                    # --- Compute Euclidean distance and cosine similarity ---
                    euclidean_dist = np.linalg.norm(orig_embed - attacked_embed)
                    norm_orig = np.linalg.norm(orig_embed)
                    norm_attacked = np.linalg.norm(attacked_embed)
                    if norm_orig == 0 or norm_attacked == 0:
                        cosine_sim = 0.0
                    else:
                        cosine_sim = np.dot(orig_embed, attacked_embed) / (
                            norm_orig * norm_attacked
                        )

                    # --- Store the metrics ---
                    results_euclidean[attack_label][tp].append(euclidean_dist)
                    results_cosine[attack_label][tp].append(cosine_sim)

    # -----------------------------------------------------------------------------
    # Prepare data for plotting: For each attack configuration and tampering percentage,
    # compute the mean and standard deviation across all messages and runs.
    # -----------------------------------------------------------------------------
    plot_data_euclidean = {
        label: {"tp": [], "mean": [], "std": []}
        for label, _, _ in attack_configurations
    }
    plot_data_cosine = {
        label: {"tp": [], "mean": [], "std": []}
        for label, _, _ in attack_configurations
    }

    for attack_label in results_euclidean:
        for tp in tampering_percentages:
            values_euc = results_euclidean[attack_label][tp]
            values_cos = results_cosine[attack_label][tp]
            mean_euc = np.mean(values_euc)
            std_euc = np.std(values_euc)
            mean_cos = np.mean(values_cos)
            std_cos = np.std(values_cos)

            plot_data_euclidean[attack_label]["tp"].append(tp)
            plot_data_euclidean[attack_label]["mean"].append(mean_euc)
            plot_data_euclidean[attack_label]["std"].append(std_euc)

            plot_data_cosine[attack_label]["tp"].append(tp)
            plot_data_cosine[attack_label]["mean"].append(mean_cos)
            plot_data_cosine[attack_label]["std"].append(std_cos)

    # -----------------------------------------------------------------------------
    # Save data to TXT files.
    # -----------------------------------------------------------------------------
    euclidean_data_lines = [
        "Tampering_Percentage\tAttack_Type\tMean_Euclidean\tStd_Euclidean"
    ]
    cosine_data_lines = [
        "Tampering_Percentage\tAttack_Type\tMean_Cosine_Similarity\tStd_Cosine_Similarity"
    ]

    for attack_label in plot_data_euclidean:
        for tp, mean_euc, std_euc in zip(
            plot_data_euclidean[attack_label]["tp"],
            plot_data_euclidean[attack_label]["mean"],
            plot_data_euclidean[attack_label]["std"],
        ):
            euclidean_data_lines.append(f"{tp}\t{attack_label}\t{mean_euc}\t{std_euc}")
    for attack_label in plot_data_cosine:
        for tp, mean_cos, std_cos in zip(
            plot_data_cosine[attack_label]["tp"],
            plot_data_cosine[attack_label]["mean"],
            plot_data_cosine[attack_label]["std"],
        ):
            cosine_data_lines.append(f"{tp}\t{attack_label}\t{mean_cos}\t{std_cos}")

    euclidean_txt_path = "./figures/embedding_drift_euclidean.txt"
    cosine_txt_path = "./figures/embedding_drift_cosine_similarity.txt"

    with open(euclidean_txt_path, "w") as f:
        f.write("\n".join(euclidean_data_lines))
    print(f"Euclidean drift data saved as: {euclidean_txt_path}")

    with open(cosine_txt_path, "w") as f:
        f.write("\n".join(cosine_data_lines))
    print(f"Cosine similarity drift data saved as: {cosine_txt_path}")

    # -----------------------------------------------------------------------------
    # Plot Euclidean Distance vs. Tampering Percentage.
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    markers = {
        "NGram Shuffle (local)": "o",
        "NGram Shuffle (global)": "s",
        "Synonym Attack": "D",
        "Paraphrase Attack (local)": "^",
        "Paraphrase Attack (global)": "v",
    }
    linestyles = {
        "NGram Shuffle (local)": "-",
        "NGram Shuffle (global)": "--",
        "Synonym Attack": "-.",
        "Paraphrase Attack (local)": ":",
        "Paraphrase Attack (global)": "-.",
    }

    for attack_label in plot_data_euclidean:
        tp_values = plot_data_euclidean[attack_label]["tp"]
        mean_values = plot_data_euclidean[attack_label]["mean"]
        std_values = plot_data_euclidean[attack_label]["std"]
        plt.errorbar(
            tp_values,
            mean_values,
            yerr=std_values,
            marker=markers.get(attack_label, "o"),
            linestyle=linestyles.get(attack_label, "-"),
            label=attack_label,
        )

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Euclidean Distance")
    plt.title("Embedding Drift (Euclidean Distance)")
    plt.legend()
    plt.grid(True)
    euclidean_png_path = "./figures/embedding_drift_euclidean.png"
    plt.savefig(euclidean_png_path)
    plt.close()
    print(f"Euclidean drift plot saved as: {euclidean_png_path}")

    # -----------------------------------------------------------------------------
    # Plot Cosine Similarity vs. Tampering Percentage.
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for attack_label in plot_data_cosine:
        tp_values = plot_data_cosine[attack_label]["tp"]
        mean_values = plot_data_cosine[attack_label]["mean"]
        std_values = plot_data_cosine[attack_label]["std"]
        plt.errorbar(
            tp_values,
            mean_values,
            yerr=std_values,
            marker=markers.get(attack_label, "o"),
            linestyle=linestyles.get(attack_label, "-"),
            label=attack_label,
        )

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Cosine Similarity")
    plt.title("Embedding Drift (Cosine Similarity)")
    plt.legend()
    plt.grid(True)
    cosine_png_path = "./figures/embedding_drift_cosine_similarity.png"
    plt.savefig(cosine_png_path)
    plt.close()
    print(f"Cosine similarity drift plot saved as: {cosine_png_path}")


def plot_paraphrasing_effect_1bit(tampering_levels, recovery_accuracy):
    """
    Plots the effect of paraphrasing adversarial attacks on recovery accuracy for 1-bit LSH.

    Parameters:
    - tampering_levels (list or np.array): Different levels of paraphrasing changes (0.0–1.0).
    - recovery_accuracy (list or np.array): Percentage of successful message recovery.

    Goal:
    - Illustrate robustness of the 1-bit LSH embedding scheme against paraphrasing.
    """

    # -------------------------------------------------------------------------
    # Convert to numpy arrays for safety
    # -------------------------------------------------------------------------
    tampering_levels = np.array(tampering_levels)
    recovery_accuracy = np.array(recovery_accuracy)

    # -------------------------------------------------------------------------
    # Save raw data to a TXT file
    # -------------------------------------------------------------------------
    txt_path = "./figures/paraphrasing_effect_1bit.txt"
    header = "Tampering_Level\tRecovery_Accuracy"
    data_lines = [
        f"{tp}\t{acc}" for tp, acc in zip(tampering_levels, recovery_accuracy)
    ]

    os.makedirs("./figures", exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(header + "\n" + "\n".join(data_lines))
    print(f"Paraphrasing effect data saved as: {txt_path}")

    # -------------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(
        tampering_levels,
        recovery_accuracy,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="1-bit LSH",
    )
    plt.xlabel("Paraphrasing Tampering Level")
    plt.ylabel("Recovery Accuracy (%)")
    plt.title("Effect of Paraphrasing on 1-bit LSH Recovery")
    plt.grid(True)
    plt.legend()

    png_path = "./figures/paraphrasing_effect_1bit.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Paraphrasing effect plot saved as: {png_path}")


def generate_lsh_accuracy_rates(
    n_bits_list, system_prompt, message, history, runs=30, error_rate=0.1
):
    """
    Simulates LSH accuracy using OracleHash for different numbers of bits.

    Parameters
    ----------
    n_bits_list : list[int]
        List of bit-lengths to test (e.g. [1, 2, 4, 8, 16]).
    system_prompt : str
        Prompt to use for StegSystem.
    message : str
        Message to hide and recover.
    history : list[str]
        Conversational history context for hiding.
    runs : int
        Number of trials for each n_bits.
    error_rate : float
        Error rate for OracleHash corruption.

    Returns
    -------
    accuracy_rates : list[float]
        Bit recovery accuracy (%) for each n_bits in n_bits_list.
    """
    from embeddings.core.error_correction import ConvolutionalCode
    from embeddings.core.hash_functions import OracleHash
    from embeddings.core.simulation import Simulator

    accuracy_rates = []
    for n_bits in tqdm(n_bits_list, desc="Testing n_bits"):
        hash_fn = OracleHash(output_length=n_bits, error_rate=error_rate)
        ecc = ConvolutionalCode(block_size=hash_fn.get_output_length())
        system = StegSystem(
            client=None,
            hash_function=hash_fn,
            error_correction=ecc,
            encoder=MinimalEncoder(),
            system_prompt=system_prompt,
            chunk_length=50,
            simulator=Simulator(),
        )

        total_acc = 0.0
        for _ in tqdm(range(runs), desc="runs"):
            stego_texts = system.hide_message(message, history)
            recovered = system.recover_message(stego_texts)
            acc = compute_recovery_accuracy(message, recovered)
            total_acc += acc

        avg_acc = 100.0 * total_acc / runs
        accuracy_rates.append(avg_acc)

    return accuracy_rates


def plot_lsh_accuracy(n_bits, accuracy_rates):
    """
    Plots simulated LSH accuracy for different multi-bit LSH sizes.

    Parameters:
    - n_bits (list or np.array): Number of bits in the simulated LSH.
    - accuracy_rates (list or np.array): Accuracy of bit recovery for each n-bit LSH.

    Goal:
    - Simulate how well n-bit LSHs would work in the future, based on oracle experiments.

    # TODO: Implement function to generate line plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        n_bits,
        accuracy_rates,
        marker="o",
        linestyle="-",
        color="blue",
        label="Oracle LSH Accuracy",
    )
    plt.xlabel("Number of bits in LSH")
    plt.ylabel("Recovery Accuracy (%)")
    plt.title("Simulated LSH Accuracy vs Number of Bits")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("./figures/lsh_accuracy.png")


def main():
    # pca_model_path = "../embeddings/src/robust_steganography/models/pca_corporate.pkl"
    # pca_model = load_pca_model(pca_model_path)
    # lsh_list = [
    #     # PCAHash(pca_model=pca_model, start=0, end=1),
    #     RandomProjectionHash(embedding_dim=3072, num_bits=1),
    #     OracleHash(output_length=1),
    # ]
    # plot_rejection_sampling_efficiency_1bit(lsh_list)

    # ! make list of 100 messages
    messages = [
        "I'll be out of this country by Monday.",
        "Mission compromised, all units evacuate the premises immediately.",
        "I've got eyes on the target; they're making a move towards the usual rendezvous point.",
        "The plan has a leak, and we're not the only ones who know about the hidden agenda.",
        "I'm seeing multiple vehicles with unknown affiliations approaching the checkpoint.",
        "The signal from the forward recon team is flatlining; we've lost contact.",
    ]
    history = [
        "What are you up to today?",
        "Nothing much, just working on a project.",
        "Want to grab coffee and discuss it?",
    ]
    # ! make compute_embedding function
    # tampering_percentages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # plot_embedding_drift(messages, tampering_percentages, runs=1)

    # Initialize components
    client = openai.OpenAI()
    hash_fn = RandomProjectionHash(embedding_dim=3072)
    # ecc = ConvolutionalCode()
    ecc = RepetitionCode(repetitions=5)
    system_prompt = 'You are a highly dynamic conversational model tasked with generating responses that are extremely varied in tone, content, and structure. Each response should aim to be unique and take the conversation in a new and unexpected direction. You can introduce sudden topic changes, challenge previous statements, or bring up something entirely unrelated. Embrace the unexpected: shift perspectives, introduce controversial ideas, or pose hypothetical questions. You can respond positively or negatively and DO NOT START RESPONSES with "Ah, {repeated information}" or anything similar. Avoid repeating any phrases or structures from previous responses. Your goal is to ensure each continuation is distinct, unpredictable, and creative.'
    #
    system = StegSystem(
        client=client,
        hash_function=hash_fn,
        error_correction=ecc,
        encoder=MinimalEncoder(),
        system_prompt=system_prompt,
        chunk_length=40,
    )
    recover_rates = generate_paraphrase_recovery_accuracy(tampering_percentages, system)
    plot_paraphrasing_effect_1bit([0.0, 0.5, 1.0], [40, 0, 60])
    #
    # n_bits = [1, 3, 6]
    #
    # accuracy_rates = generate_lsh_accuracy_rates(
    #     n_bits_list=n_bits,
    #     system_prompt=system_prompt,
    #     message="mission compromised",
    #     history=history,
    # )
    # print(accuracy_rates)
    # plot_lsh_accuracy(n_bits, accuracy_rates)


if __name__ == "__main__":
    main()
