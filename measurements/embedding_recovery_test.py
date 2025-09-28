import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import openai
from robust_steganography.core.encoder import Encoder
from robust_steganography.core.error_correction import RepetitionCode
from robust_steganography.core.hash_functions import (
    RandomProjectionHash,
)
from robust_steganography.core.steg_system import StegSystem
from tqdm import tqdm
from watermark import GPT2Model
from watermark.attacks.ngram_shuffle import NGramShuffleAttack
from watermark.attacks.paraphrase import ParaphraseAttack
from watermark.attacks.synonym import SynonymAttack
from watermark.models.base import LanguageModel

from watermarks import (
    apply_partial_paraphrase,
    compute_recovery_accuracy,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Helpers
class BypassEncoder(Encoder):
    def __init__(self):
        pass

    @override
    def encode(self, data: list[int]) -> list[int]:
        return data
    
    @override
    def decode(self, bits: list[int]) -> list[int]:
        return bits

def attack(
    attack_type: str,
    messages: list[str],
    paraphrase_instance: ParaphraseAttack,
    model: LanguageModel,
    tp: float,
    mode: bool,
) -> list[str]:
    attacked_texts = []
    for mes in messages:
        if attack_type == "n-gram":
            attack = NGramShuffleAttack(model=model, n=3, probability=tp, local=mode)
            attacked_texts.append(attack(mes))
        elif attack_type == "synonym":
            attack = SynonymAttack(method="wordnet", probability=tp)
            attacked_texts.append(attack(mes))
        elif attack_type == "paraphrase":
            if tp < 1.0:
                attacked_text = apply_partial_paraphrase(mes, paraphrase_instance, tp)
            else:
                attacked_text = paraphrase_instance(mes)
            attacked_texts.append(attacked_text)
        else:
            print("Unsupported attack mode.")
            attacked_texts.append(None)

    return attacked_texts

def save_pickle(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def append_lines(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def generate_recovery_accuracy_resumable(
    tampering_levels: list[float],
    attack_configurations: list,
    system: StegSystem,
    num_bits: int,
    num_messages: int,
    history: list[str] | None = None,
    runs: int = 5,
    seed: int | None = None,
    checkpoint_path: str = "checkpoints/exp_checkpoint.pkl",
    output_path: str = "exp_results",
    save_texts: bool = False,
    max_saved_examples: int = 200,
    resume: bool = True,
    checkpoint_after_each_message: bool = False,
):
    """
    Resumable, logged version of generate_recovery_accuracy.

    Parameters:
    - tampering_levels: iterable of tampering percentages
    - system: object with hide_message(message, history) and recover_message(attacked_texts)
              and encoder.encode(...)
    - seed: optional random seed for deterministic message generation
    - checkpoint_path: pickle file path for checkpointing/resume
    - out_dir: directory to write CSV/JSONL logs
    - save_texts: whether to save stego/attacked/recovered text outputs (can be large)
    - max_saved_examples: maximum number of examples to persist across the whole run (to limit size)
    - resume: if True and checkpoint exists, resume from it
    - checkpoint_after_each_message: if True, checkpoint more frequently (slower)
    """

    out_dir = Path(output_path)
    checkpoint_file = Path(checkpoint_path)
    texts_log_path = out_dir / "texts_log.jsonl"
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)


    if seed is not None:
        random.seed(seed)

    if history is None:
        history = [
            "What are you up to today?",
            "Nothing much, just working on a project.",
            "Want to grab coffee and discuss it?",
        ]

    paraphrase_instance = ParaphraseAttack(
        client=system.client,
        model="gpt-4.1-nano",
        temperature=0.0,
    )
    model = GPT2Model()

    messages: list[list[int]] = [[random.randint(0, 1) for _ in range(num_bits)] for _ in range(num_messages)]

    if resume and checkpoint_file.exists():
        checkpoint = load_pickle(checkpoint_file)
        LOGGER.info(f"[resume] loaded checkpoint from {checkpoint_file}")
    else:
        checkpoint = {
            "attack_index": 0,
            "tampering_index": 0,
            "message_index": 0,
            "run_index": 0,
            "all_ret": {},  # store final results per attack label
            "texts_saved_count": 0,
            "random_state": None,
        }

    # Get random state for resume reseeding
    checkpoint["random_state"] = random.getstate()

    save_pickle(checkpoint_file, checkpoint)

    all_ret = checkpoint.get("all_ret", {})

    LOGGER.info("Start Experiment")
    for a_idx in range(checkpoint["attack_index"], len(attack_configurations)):
        attack_label, attack_type, mode = attack_configurations[a_idx]
        mode_k = "None" if mode is None else str(mode)

        LOGGER.info(f"Starting attack {a_idx+1}/{len(attack_configurations)}: {attack_label} ({attack_type}, mode={mode})")

        if attack_label in all_ret:
            results_bitwise = all_ret[attack_label].get("results_bitwise", {mode_k: []})
            results_perfect = all_ret[attack_label].get("results_perfect", {mode_k: []})
            data_lines_bitwise = all_ret[attack_label].get("data_lines_bitwise", ["Tampering_Percentage\tMode\tBitwise_Accuracy"])
            data_lines_perfect = all_ret[attack_label].get("data_lines_perfect", ["Tampering_Percentage\tMode\tPerfect_Recovery_Rate"])
        else:
            results_bitwise = {mode_k: []}
            results_perfect = {mode_k: []}
            data_lines_bitwise = ["Tampering_Percentage\tMode\tBitwise_Accuracy"]
            data_lines_perfect = ["Tampering_Percentage\tMode\tPerfect_Recovery_Rate"]

        # iterate tampering levels
        tampering_pbar = tqdm(
            tampering_levels,
            desc=f"Tampering levels ({attack_label})",
            position=0,
            leave=True,
            initial=checkpoint["tampering_index"],
        )
        for t_idx, tp in enumerate(tampering_levels[checkpoint["tampering_index"]:],
                                   start=checkpoint["tampering_index"]):
            tampering_pbar.n = t_idx  # keep position consistent
            tampering_pbar.refresh()
            tp = tampering_levels[t_idx]

            perfect_recovery = 0.0
            bitwise_recovery = 0.0

            msg_pbar = tqdm(
                range(len(messages)),
                desc=f"Messages@tp={tp}",
                position=1,
                leave=False,
                initial=checkpoint["message_index"],
            )
            for m_idx in range(checkpoint["message_index"], len(messages)):
                msg_pbar.update(1)
                message = messages[m_idx]
                success_count = 0
                bit_score = 0.0

                run_pbar = tqdm(
                    range(runs),
                    desc=f"runs (msg {m_idx}, tp={tp})",
                    position=2,
                    leave=False,
                    initial=checkpoint.get("run_index", 0),
                )

                for run_i in range(checkpoint.get("run_index", 0), runs):
                    run_pbar.update(1)
                    stego_texts = system.hide_message(message, history)
                    attacked_texts = attack(
                        attack_type=attack_type,
                        messages=stego_texts,
                        paraphrase_instance=paraphrase_instance,
                        model=model,
                        tp=tp,
                        mode=mode,
                    )
                    recovered = system.recover_message(attacked_texts)

                    # perfect recovery
                    if recovered == message:
                        success_count += 1

                    # bitwise score
                    encoded_truth = system.encoder.encode(message)
                    encoded_rec = system.encoder.encode(recovered)
                    if encoded_truth is None or encoded_rec is None:
                        this_bitwise = 0.0
                    else:
                        this_bitwise = compute_recovery_accuracy(encoded_truth, encoded_rec)
                    bit_score += this_bitwise

                    # optionally save texts for debugging (limited)
                    if save_texts and checkpoint["texts_saved_count"] < max_saved_examples:
                        record = {
                            "timestamp": time.time(),
                            "attack_label": attack_label,
                            "attack_type": attack_type,
                            "mode": mode_k,
                            "tampering": tp,
                            "message_index": m_idx,
                            "run_index": run_i,
                            "message_bits": message,
                            "stego_texts": stego_texts,
                            "attacked_texts": attacked_texts,
                            "recovered": recovered,
                        }
                        # append as JSON line
                        with open(texts_log_path, "a", encoding="utf-8") as tf:
                            tf.write(json.dumps(record, default=str) + "\n")
                        checkpoint["texts_saved_count"] += 1

                    # update run-level checkpoint indices so we can resume precisely
                    checkpoint["attack_index"] = a_idx
                    checkpoint["tampering_index"] = t_idx
                    checkpoint["message_index"] = m_idx
                    checkpoint["run_index"] = run_i + 1
                    save_pickle(checkpoint_file, checkpoint)

                run_pbar.close()
                # end runs for message
                perfect_recovery_run = success_count / float(runs) if runs > 0 else 0.0
                bitwise_recovery_run = bit_score / float(runs) if runs > 0 else 0.0
                perfect_recovery += perfect_recovery_run
                bitwise_recovery += bitwise_recovery_run

                # message-level checkpoint: reset run_index to 0 for next message
                checkpoint["run_index"] = 0
                checkpoint["message_index"] = m_idx + 1
                if checkpoint_after_each_message:
                    save_pickle(checkpoint_file, checkpoint)

            msg_pbar.close()
            # end messages loop
            perfect = 100.0 * perfect_recovery / float(len(messages))
            bitwise = 100.0 * bitwise_recovery / float(len(messages))

            results_bitwise[mode_k].append(bitwise)
            results_perfect[mode_k].append(perfect)
            data_lines_bitwise.append(f"{tp}\t{mode_k}\t{bitwise}")
            data_lines_perfect.append(f"{tp}\t{mode_k}\t{perfect}")

            # save summary CSV/TSV for this attack so partial outputs are available
            tsv_bitwise_path = summary_dir / f"{attack_label.replace(' ', '_')}_bitwise.tsv"
            tsv_perfect_path = summary_dir / f"{attack_label.replace(' ', '_')}_perfect.tsv"
            # write full file (overwrite) so it always contains complete so far results
            with open(tsv_bitwise_path, "w", encoding="utf-8") as f:
                f.write("\n".join(data_lines_bitwise) + "\n")
            with open(tsv_perfect_path, "w", encoding="utf-8") as f:
                f.write("\n".join(data_lines_perfect) + "\n")

            # update all_ret and checkpoint
            all_ret[attack_label] = {
                "results_bitwise": results_bitwise,
                "results_perfect": results_perfect,
                "data_lines_bitwise": data_lines_bitwise,
                "data_lines_perfect": data_lines_perfect,
            }
            checkpoint["all_ret"] = all_ret

            checkpoint["tampering_index"] = t_idx + 1
            checkpoint["message_index"] = 0
            checkpoint["run_index"] = 0

            # persist checkpoint after each tampering level (cheap)
            save_pickle(checkpoint_file, checkpoint)

        tampering_pbar.close()
        # end tampering_levels loop
        checkpoint["tampering_index"] = 0
        checkpoint["message_index"] = 0
        checkpoint["run_index"] = 0
        checkpoint["attack_index"] = a_idx + 1
        checkpoint["all_ret"] = all_ret
        save_pickle(checkpoint_file, checkpoint)

    LOGGER.info("Experiment finished; final results saved into checkpoint and summary directory.")
    return all_ret

def generate_recovery_accuracy(
    tampering_levels: list,
    system: StegSystem,
    num_bits: int,
    num_messages: int,
    history: list[str] | None = None,
    runs: int = 5,
):
    import random
    if history is None:
        history = [
            "What are you up to today?",
            "Nothing much, just working on a project.",
            "Want to grab coffee and discuss it?",
        ]
    attack_configurations = [
        # ("NGram Shuffle (local)", "n-gram", True),
        # ("NGram Shuffle (global)", "n-gram", False),
        # ("Synonym Attack", "synonym", None),
        ("Paraphrase Attack (local)", "paraphrase", True),
        ("Paraphrase Attack (global)", "paraphrase", False),
    ]
    paraphrase_instance = ParaphraseAttack(
        client=system.client,
        model="gpt-5-nano",
        temperature=0.0,
    )
    model = GPT2Model()

    # construct random secret messages using num_bits and num_messages
    messages: list[list[int]] = [[random.randint(0, 1) for _ in range(num_bits)] for _ in range(num_messages)]

    all_ret = {}
    for attack_label, attack_type, mode in attack_configurations:
        LOGGER.info(f"Testing {attack_label}, {attack_type}, {mode}")
        results_bitwise = {mode: []}
        results_perfect = {mode: []}
        data_lines_bitwise = ["Tampering_Percentage\tMode\tBitwise_Accuracy"]
        data_lines_perfect = ["Tampering_Percentage\tMode\tPerfect_Recovery_Rate"]
        for tp in tqdm(tampering_levels, desc="Tampering levels"):
            bitwise_recovery = 0
            perfect_recovery = 0
            for message in tqdm(messages, desc="Messages"):
                success_count = 0
                bit_score = 0

                # run trials
                for _ in tqdm(range(runs), desc=f"runs@tp={tp}", leave=False):
                    try:
                        stego_texts = system.hide_message(message, history)
                    except Exception as e:
                        LOGGER.warning(f"system.hide_message failed: {e}")
                        continue
                    attacked_texts = attack(
                        attack_type=attack_type,
                        messages=stego_texts,
                        paraphrase_instance=paraphrase_instance,
                        model=model,
                        tp=tp,
                        mode=mode,
                    )

                    # perfect recovery score
                    try:
                        recovered = system.recover_message(attacked_texts)
                    except Exception:
                        LOGGER.warning("Message recovery failure")
                        recovered = None
                    if recovered == message:
                        success_count += 1
                    # bit-wise score
                    bit_score += compute_recovery_accuracy(
                        system.encoder.encode(message), system.encoder.encode(recovered)
                    )

                perfect_recovery_run = (
                    success_count / float(runs) if runs > 0 else 0.0
                )
                bitwise_recovery_run = (
                    bit_score / float(runs) if runs > 0 else 0.0
                )
                perfect_recovery += perfect_recovery_run
                bitwise_recovery += bitwise_recovery_run

            perfect = perfect_recovery / float(num_messages)
            bitwise = bitwise_recovery /float(num_messages)

            results_bitwise[mode].append(bitwise)
            results_perfect[mode].append(perfect)
            data_lines_bitwise.append(f"{tp}\t{mode}\t{bitwise}")
            data_lines_perfect.append(f"{tp}\t{mode}\t{perfect}")

        all_ret[attack_label] = (
            results_bitwise,
            results_perfect,
            data_lines_bitwise,
            data_lines_perfect,
        )

    return all_ret

def plot_recovery_results(tp, attack_types, results, output_path):
    for attack_type in attack_types:
        results_bitwise = results[attack_type]['results_bitwise']
        results_perfect = results[attack_type]['results_perfect']
        datalines_bitwise = results[attack_type]['data_lines_bitwise']
        data_lines_perfect = results[attack_type]['data_lines_perfect']

        plt.figure(figsize=(8, 6))
        # plt.axhline(baseline_recovery, color='gray', linestyle='--', label="Baseline")
        markers = {"local": "o", "global": "s"}
        linestyles = {"local": "-", "global": "--"}
        for mode_label, acc_list in results_bitwise.items():
            plt.plot(
                tp,
                acc_list,
                marker=markers.get(mode_label, "o"),
                linestyle=linestyles.get(mode_label, "-"),
                label=mode_label,
            )
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
        with open(txt_filename_bitwise, "w") as f:
            f.write("\n".join(datalines_bitwise))
        print(f"Bitwise data saved as: {txt_filename_bitwise}")

        # -----------------------------------------------------------------------------
        # Plot Perfect Recovery Rate for this attack type.
        # -----------------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        # plt.axhline(1.0, color="gray", linestyle="--", label="100% Perfect Recovery")
        for mode_label, rate_list in results_perfect.items():
            plt.plot(
                tp,
                rate_list,
                marker=markers.get(mode_label, "o"),
                linestyle=linestyles.get(mode_label, "-"),
                label=mode_label,
            )
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
        with open(txt_filename_perfect, "w") as f:
            f.write("\n".join(data_lines_perfect))
        print(f"Perfect recovery data saved as: {txt_filename_perfect}")


def main():
    from robust_steganography.config.system_prompts import CORPORATE_MONOLOGUE
    tp = [0.1, 0.5]

    # Initialize components
    client = openai.OpenAI()
    # hash_fn = PCAHash(
    #     pca_model=load_pca_model("../embeddings/src/robust_steganography/models/pca_corporate.pkl")
    # )
    hash_fn = RandomProjectionHash(embedding_dim=3072)
    
    # ecc = ConvolutionalCode()
    ecc = RepetitionCode(repetitions=5)
    system_prompt = CORPORATE_MONOLOGUE

    history = [
        "I wanted to follow up regarding the implementation timeline for the new risk management system. Based on our initial assessment, we'll need to coordinate closely with both IT and Operations to ensure a smooth transition. Please review the attached documentation when you have a moment.",
        "After consulting with the development team, we've identified several key milestones that need to be addressed before proceeding. The current testing phase has revealed some potential integration issues with our legacy systems, particularly in the trade validation module. We're working on implementing the necessary fixes and expect to have an updated timeline by end of week.",
        "Given the complexity of these changes, I believe it would be beneficial to schedule a stakeholder review meeting. We should include representatives from Risk Management, IT Operations, and the Trading desk to ensure all requirements are being met. I've asked Sarah to coordinate calendars for next Tuesday afternoon.",
    ]
    system = StegSystem(
        client=client,
        hash_function=hash_fn,
        error_correction=ecc,
        encoder=BypassEncoder(),
        system_prompt=system_prompt,
        chunk_length=40,
    )

    attack_configurations = [
        # ("NGram Shuffle (local)", "n-gram", True),
        # ("NGram Shuffle (global)", "n-gram", False),
        # ("Synonym Attack", "synonym", None),
        ("Paraphrase Attack (local)", "paraphrase", True),
        ("Paraphrase Attack (global)", "paraphrase", False),
    ]

    attack_keys = [t[0] for t in attack_configurations]
    results = generate_recovery_accuracy_resumable(
        tampering_levels=tp,
        attack_configurations = attack_configurations,
        system=system, 
        num_bits=1,
        num_messages=1,
        history=history,
        runs=3,
        seed=42,
        checkpoint_path="checkpoints/exp_checkpoint.pkl",
        output_path="figures/embedding_recovery_test",
        save_texts=False,
        max_saved_examples=200,
        resume=True,
        checkpoint_after_each_message=False,
    )
    print(results)

    output_path = "./figures/embedding_recovery_test/"
    plot_recovery_results(tp,attack_keys, results, output_path)


if __name__ == "__main__":
    main()
