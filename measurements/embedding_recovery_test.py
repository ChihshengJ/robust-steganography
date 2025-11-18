import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import openai

# from src.embeddings.utils.get_embedding import client
from tqdm import tqdm

from embeddings import Encoder, RandomProjectionHash, RepetitionCode, StegSystem
from embeddings.config.system_prompts import STORY_GENERATION
from watermarks import (
    Attack,
    GPT2Model,
    LanguageModel,
    NGramShuffleAttack,
    ParaphraseAttack,
    SynonymAttack,
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
    at: dict[str, Attack],
    messages: list[str],
    tp: float,
    local: bool,
) -> str:
    # print(f"attack input message: {messages}")
    stego_text = " ".join(messages)
    print(f"======stego_text======\n{stego_text}")
    result = at[attack_type](stego_text, tp, local)
    print(f"======attacked_text======\n{result}")
    return result


def init_attacks(model: LanguageModel, client) -> dict[str, Attack]:
    return {
        "n-gram": NGramShuffleAttack(model=model, n=3),
        "synonym": SynonymAttack(method="wordnet"),
        "paraphrase": ParaphraseAttack(
            client=client, model="gpt-4o-mini", temperature=0
        ),
    }


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


def compute_recovery_accuracy(original, recovered):
    """
    Computes bitwise accuracy between two lists of bits.
    Returns the fraction of matching bits.
    """
    if len(original) != len(recovered):
        return 0.0
    correct = sum(1 for o, r in zip(original, recovered) if o == r)
    return correct / len(original)


# Experiment
def generate_recovery_accuracy_resumable(
    tampering_levels: list[float],
    attack_configurations: list,
    system: StegSystem,
    num_bits: int,
    num_messages: int,
    num_stego_per_message: int,
    runs: int = 5,
    history: list[str] | None = None,
    seed: int | None = None,
    checkpoint_path: str = "checkpoints/exp_checkpoint.pkl",
    output_path: str = "exp_results",
    save_texts: bool = False,
    max_saved_examples: int = 200,
    resume: bool = True,
    checkpoint_after_each_stego: bool = False,
):
    """
    Loop structure:
    1. For each message:
       a. Generate num_stego_per_message stego texts
       b. For each attack configuration:
          For each tampering level:
            For each stego text:
              For each run:
                Attack and recover
              Aggregate across runs
            Aggregate across stego texts
          Aggregate across tampering levels
       c. Aggregate across attacks
    2. Aggregate across all messages

    Parameters:
    - tampering_levels: iterable of tampering percentages
    - attack_configurations: list of (label, type, config) tuples
    - system: StegSystem with hide_message() and recover_message()
    - num_bits: number of bits per message
    - num_messages: number of different messages to test
    - num_stego_per_message: number of stego texts to generate per message
    - runs: number of attack/recovery runs per stego text
    - history: conversation history for stego generation
    - seed: optional random seed for deterministic message generation
    - checkpoint_path: pickle file path for checkpointing/resume
    - output_path: directory to write CSV/JSONL logs
    - save_texts: whether to save stego/attacked/recovered text outputs
    - max_saved_examples: maximum number of examples to persist
    - resume: if True and checkpoint exists, resume from it
    - checkpoint_after_each_stego: if True, checkpoint after each stego generation
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

    # paraphrase_instance = ParaphraseAttack(
    #     client=system.client, model="gpt-4.1-nano", temperature=0.5, local=False
    # )
    model = GPT2Model()

    messages: list[list[int]] = [
        [random.randint(0, 1) for _ in range(num_bits)] for _ in range(num_messages)
    ]
    print(f"messages: {messages}")

    if resume and checkpoint_file.exists():
        checkpoint = load_pickle(checkpoint_file)
        LOGGER.info(f"[resume] loaded checkpoint from {checkpoint_file}")
    else:
        checkpoint = {
            "message_index": 0,
            "current_message_stego_complete": False,
            "stego_gen_index": 0,
            "attack_index": 0,
            "tampering_index": 0,
            "stego_index": 0,
            "run_index": 0,
            "all_ret": {},
            "message_results": {},
            "texts_saved_count": 0,
            "random_state": None,
            "current_stego_texts": [],
        }

    checkpoint["random_state"] = random.getstate()
    save_pickle(checkpoint_file, checkpoint)

    all_ret = checkpoint.get("all_ret", {})
    LOGGER.info("Start Experiment")

    # Main Loop
    for msg_idx in range(checkpoint["message_index"], num_messages):
        message = messages[msg_idx]
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"Processing Message {msg_idx + 1}/{num_messages}")
        LOGGER.info(f"{'=' * 60}")

        # Generating stegotexts
        if not checkpoint.get("current_message_stego_complete", False):
            LOGGER.info(
                f"Generating {num_stego_per_message} stego texts for message {msg_idx}"
            )
            current_stego_texts = checkpoint.get("current_stego_texts", [])

            stego_gen_pbar = tqdm(
                total=num_stego_per_message,
                desc=f"Generating stego (msg {msg_idx})",
                initial=checkpoint["stego_gen_index"],
            )

            for stego_i in range(checkpoint["stego_gen_index"], num_stego_per_message):
                stego_texts = system.hide_message(message, history)
                current_stego_texts.append(stego_texts)

                checkpoint["stego_gen_index"] = stego_i + 1
                checkpoint["current_stego_texts"] = current_stego_texts

                if checkpoint_after_each_stego:
                    save_pickle(checkpoint_file, checkpoint)

                stego_gen_pbar.update(1)

            stego_gen_pbar.close()

            checkpoint["current_message_stego_complete"] = True
            checkpoint["current_stego_texts"] = current_stego_texts
            save_pickle(checkpoint_file, checkpoint)
            LOGGER.info(f"Stego generation complete for message {msg_idx}")
        else:
            current_stego_texts = checkpoint["current_stego_texts"]
            LOGGER.info(
                f"Using {len(current_stego_texts)} pre-generated stego texts for message {msg_idx}"
            )

        # Start attacks
        message_results = checkpoint.get("message_results", {})
        at = init_attacks(model=model, client=system.client)
        for a_idx in range(checkpoint["attack_index"], len(attack_configurations)):
            attack_label, attack_type, mode = attack_configurations[a_idx]
            LOGGER.info(
                f"\nAttack {a_idx + 1}/{len(attack_configurations)}: {attack_label, mode} (Message {msg_idx})"
            )
            if attack_label not in all_ret:
                all_ret[attack_label] = {
                    "results_bitwise": [],
                    "results_perfect": [],
                    "data_lines_bitwise": ["Tampering_Percentage\tBitwise_Accuracy"],
                    "data_lines_perfect": [
                        "Tampering_Percentage\tPerfect_Recovery_Rate"
                    ],
                    "tampering_level_data": {},
                }
            if attack_label not in message_results:
                message_results[attack_label] = {"tampering_level_data": {}}
            tampering_pbar = tqdm(
                tampering_levels,
                desc=f"Tampering ({attack_label}, msg {msg_idx})",
                position=0,
                leave=True,
                initial=checkpoint["tampering_index"],
            )
            for t_idx in range(checkpoint["tampering_index"], len(tampering_levels)):
                tampering_pbar.n = t_idx
                tampering_pbar.refresh()
                tp = tampering_levels[t_idx]
                if tp not in message_results[attack_label]["tampering_level_data"]:
                    message_results[attack_label]["tampering_level_data"][tp] = {
                        "perfect_scores": [],
                        "bitwise_scores": [],
                    }
                stego_pbar = tqdm(
                    range(len(current_stego_texts)),
                    desc=f"Stego texts@tp={tp}",
                    position=1,
                    leave=False,
                    initial=checkpoint["stego_index"],
                )
                for s_idx in range(checkpoint["stego_index"], len(current_stego_texts)):
                    stego_pbar.update(1)
                    stego_texts = current_stego_texts[s_idx]
                    success_count = 0
                    bit_score = 0.0
                    run_pbar = tqdm(
                        range(runs),
                        desc=f"Runs (stego {s_idx})",
                        position=2,
                        leave=False,
                        initial=checkpoint.get("run_index", 0),
                    )
                    for run_i in range(checkpoint.get("run_index", 0), runs):
                        run_pbar.update(1)
                        attacked_text = attack(
                            attack_type=attack_type,
                            at=at,
                            messages=stego_texts,
                            tp=tp,
                            local=mode,
                        )
                        # print(f"===========\nattacked text:\n{attacked_text}")
                        system.set_chunk_length(len(stego_texts))
                        recovered = system.recover_message(attacked_text)
                        print(f"message: {message}, recovered: {recovered}")
                        if recovered == message:
                            success_count += 1
                        encoded_truth = system.encoder.encode(message)
                        encoded_rec = system.encoder.encode(recovered)
                        if encoded_truth is None or encoded_rec is None:
                            this_bitwise = 0.0
                        else:
                            this_bitwise = compute_recovery_accuracy(
                                encoded_truth, encoded_rec
                            )
                        bit_score += this_bitwise

                        if (
                            save_texts
                            and checkpoint["texts_saved_count"] < max_saved_examples
                        ):
                            record = {
                                "timestamp": time.time(),
                                "attack_label": attack_label,
                                "attack_type": attack_type,
                                "tampering": tp,
                                "message_index": msg_idx,
                                "stego_index": s_idx,
                                "run_index": run_i,
                                "message_bits": message,
                                "stego_texts": stego_texts,
                                "attacked_texts": attacked_text,
                                "recovered": recovered,
                            }
                            with open(texts_log_path, "a", encoding="utf-8") as tf:
                                tf.write(json.dumps(record, default=str) + "\n")
                            checkpoint["texts_saved_count"] += 1

                        checkpoint["message_index"] = msg_idx
                        checkpoint["attack_index"] = a_idx
                        checkpoint["tampering_index"] = t_idx
                        checkpoint["stego_index"] = s_idx
                        checkpoint["run_index"] = run_i + 1
                        save_pickle(checkpoint_file, checkpoint)

                    run_pbar.close()

                    perfect_for_stego = success_count / float(runs) if runs > 0 else 0.0
                    bitwise_for_stego = bit_score / float(runs) if runs > 0 else 0.0

                    message_results[attack_label]["tampering_level_data"][tp][
                        "perfect_scores"
                    ].append(perfect_for_stego)
                    message_results[attack_label]["tampering_level_data"][tp][
                        "bitwise_scores"
                    ].append(bitwise_for_stego)

                    checkpoint["run_index"] = 0
                    checkpoint["stego_index"] = s_idx + 1
                    checkpoint["message_results"] = message_results
                    save_pickle(checkpoint_file, checkpoint)

                stego_pbar.close()

                tp_data = message_results[attack_label]["tampering_level_data"][tp]
                perfect_avg = (
                    sum(tp_data["perfect_scores"]) / len(tp_data["perfect_scores"])
                    if tp_data["perfect_scores"]
                    else 0.0
                )
                bitwise_avg = (
                    sum(tp_data["bitwise_scores"]) / len(tp_data["bitwise_scores"])
                    if tp_data["bitwise_scores"]
                    else 0.0
                )

                if tp not in all_ret[attack_label]["tampering_level_data"]:
                    all_ret[attack_label]["tampering_level_data"][tp] = {
                        "perfect_per_message": [],
                        "bitwise_per_message": [],
                    }

                all_ret[attack_label]["tampering_level_data"][tp][
                    "perfect_per_message"
                ].append(perfect_avg)
                all_ret[attack_label]["tampering_level_data"][tp][
                    "bitwise_per_message"
                ].append(bitwise_avg)

                checkpoint["stego_index"] = 0
                checkpoint["run_index"] = 0
                checkpoint["tampering_index"] = t_idx + 1
                checkpoint["all_ret"] = all_ret
                save_pickle(checkpoint_file, checkpoint)

            tampering_pbar.close()

            checkpoint["tampering_index"] = 0
            checkpoint["stego_index"] = 0
            checkpoint["run_index"] = 0
            checkpoint["attack_index"] = a_idx + 1
            checkpoint["all_ret"] = all_ret
            checkpoint["message_results"] = message_results
            save_pickle(checkpoint_file, checkpoint)

        checkpoint["message_index"] = msg_idx + 1
        checkpoint["attack_index"] = 0
        checkpoint["tampering_index"] = 0
        checkpoint["stego_index"] = 0
        checkpoint["run_index"] = 0
        checkpoint["current_message_stego_complete"] = False
        checkpoint["stego_gen_index"] = 0
        checkpoint["current_stego_texts"] = []
        checkpoint["message_results"] = {}
        save_pickle(checkpoint_file, checkpoint)

    # Aggregating results
    LOGGER.info("\nFinal aggregation across all messages")

    for attack_label in all_ret:
        results_bitwise = []
        results_perfect = []
        data_lines_bitwise = ["Tampering_Percentage\tBitwise_Accuracy"]
        data_lines_perfect = ["Tampering_Percentage\tPerfect_Recovery_Rate"]

        for tp in tampering_levels:
            tp_data = all_ret[attack_label]["tampering_level_data"][tp]

            perfect_final = (
                100.0
                * sum(tp_data["perfect_per_message"])
                / len(tp_data["perfect_per_message"])
            )
            bitwise_final = (
                100.0
                * sum(tp_data["bitwise_per_message"])
                / len(tp_data["bitwise_per_message"])
            )

            results_bitwise.append(bitwise_final)
            results_perfect.append(perfect_final)
            data_lines_bitwise.append(f"{tp}\t{bitwise_final}")
            data_lines_perfect.append(f"{tp}\t{perfect_final}")

        all_ret[attack_label]["results_bitwise"] = results_bitwise
        all_ret[attack_label]["results_perfect"] = results_perfect
        all_ret[attack_label]["data_lines_bitwise"] = data_lines_bitwise
        all_ret[attack_label]["data_lines_perfect"] = data_lines_perfect

        tsv_bitwise_path = summary_dir / f"{attack_label.replace(' ', '_')}_bitwise.tsv"
        tsv_perfect_path = summary_dir / f"{attack_label.replace(' ', '_')}_perfect.tsv"

        with open(tsv_bitwise_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data_lines_bitwise) + "\n")
        with open(tsv_perfect_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data_lines_perfect) + "\n")

    checkpoint["all_ret"] = all_ret
    save_pickle(checkpoint_file, checkpoint)

    LOGGER.info("Experiment finished; final results saved.")
    return all_ret


def plot_recovery_results(tp, attack_types, results, output_path):
    # -------------------------------------------------------------------------
    # Bitwise Recovery Accuracy (all attack types in one plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for attack_type in attack_types:
        results_bitwise = results[attack_type]["results_bitwise"]
        datalines_bitwise = results[attack_type]["data_lines_bitwise"]
        attack_fname = attack_type.lower().replace(" ", "_")

        plt.plot(tp, results_bitwise, label=f"{attack_type}")

        # save the corresponding txt data file per attack
        txt_filename_bitwise = f"{output_path}{attack_fname}_bitwise.txt"
        with open(txt_filename_bitwise, "w") as f:
            f.write("\n".join(datalines_bitwise))
        print(f"Bitwise data saved as: {txt_filename_bitwise}")

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Bitwise Recovery Accuracy")
    plt.title("Bitwise Recovery Accuracy Across Attacks")
    plt.legend()
    plt.grid(True)
    png_filename_bitwise = f"{output_path}all_attacks_bitwise.png"
    plt.savefig(png_filename_bitwise)
    plt.close()
    print(f"Bitwise graph saved as: {png_filename_bitwise}")

    # -------------------------------------------------------------------------
    # Perfect Recovery Rate (all attack types in one plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for attack_type in attack_types:
        results_perfect = results[attack_type]["results_perfect"]
        data_lines_perfect = results[attack_type]["data_lines_perfect"]
        attack_fname = attack_type.lower().replace(" ", "_")

        plt.plot(tp, results_perfect, label=f"{attack_type}")

        # save the corresponding txt data file per attack
        txt_filename_perfect = f"{output_path}{attack_fname}_perfect.txt"
        with open(txt_filename_perfect, "w") as f:
            f.write("\n".join(data_lines_perfect))
        print(f"Perfect recovery data saved as: {txt_filename_perfect}")

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Perfect Recovery Rate")
    plt.title("Perfect Recovery Rate Across Attacks")
    plt.legend()
    plt.grid(True)
    png_filename_perfect = f"{output_path}all_attacks_perfect.png"
    plt.savefig(png_filename_perfect)
    plt.close()
    print(f"Perfect recovery graph saved as: {png_filename_perfect}")


def main():
    # tp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tp = [0.5, 1.0]

    # Initialize components
    client = openai.OpenAI()
    # hash_fn = PCAHash(
    #     pca_model=load_pca_model("../embeddings/src/robust_steganography/models/pca_corporate.pkl")
    # )
    hash_fn = RandomProjectionHash(embedding_dim=3072, seed=42)

    ##############################################################################
    # The length of the stegotext is controlled by the error correction algorithm:
    # Repetition: reprtition * num_bits
    # Convolution: 4 * (num_bits + K -1)
    ##############################################################################

    # ecc = ConvolutionalCode(block_size=1, K=3)
    ecc = RepetitionCode(5)
    system_prompt = STORY_GENERATION.format(
        items="kids, ice cream, truck",
        boring_theme="Kids getting ice cream from an ice cream truck",
    )
    # system_prompt = CORPORATE_MONOLOGUE_ALT

    # history = [
    #     "I wanted to follow up regarding the implementation timeline for the new risk management system. Based on our initial assessment, we'll need to coordinate closely with both IT and Operations to ensure a smooth transition. Please review the attached documentation when you have a moment.",
    #     "After consulting with the development team, we've identified several key milestones that need to be addressed before proceeding. The current testing phase has revealed some potential integration issues with our legacy systems, particularly in the trade validation module. We're working on implementing the necessary fixes and expect to have an updated timeline by end of week.",
    #     "Given the complexity of these changes, I believe it would be beneficial to schedule a stakeholder review meeting. We should include representatives from Risk Management, IT Operations, and the Trading desk to ensure all requirements are being met. I've asked Sarah to coordinate calendars for next Tuesday afternoon.",
    # ]
    history = []
    system = StegSystem(
        client=client,
        hash_function=hash_fn,
        error_correction=ecc,
        encoder=BypassEncoder(),
        system_prompt=system_prompt,
        max_length=200,
        story_mode=True,
    )

    # Right now we treat tp == 1 as global so there is no need to tamper with mode
    attack_configurations = [
        # ("NGram Shuffle", "n-gram", True),
        # ("Synonym Attack", "synonym", None),
        ("Paraphrase Attack", "paraphrase", True),
        # ("Translate Attack", "translate", True),
    ]

    attack_keys = [t[0] for t in attack_configurations]

    params = {
        "tampering_levels": tp,
        "attack_configurations": attack_configurations,
        "system": system,
        "num_bits": 3,
        "num_messages": 1,
        "num_stego_per_message": 2,
        "runs": 10,
        "history": history,
        "seed": 8485,
        "checkpoint_path": "checkpoints/test/exp_checkpoint.pkl",
        "output_path": "figures/test/embedding_recovery_test",
        "save_texts": True,
        "max_saved_examples": 200,
        "resume": True,
        "checkpoint_after_each_stego": True,
    }

    results = generate_recovery_accuracy_resumable(**params)
    print(results)

    # output_path = "./figures/embedding_recovery_test/"
    # plot_recovery_results(tp, attack_keys, results, output_path)


if __name__ == "__main__":
    main()
