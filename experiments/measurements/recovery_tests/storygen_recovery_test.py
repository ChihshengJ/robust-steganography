import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import openai

from embeddings.paths import pca_artifacts_dir
from embeddings import (
    PCAHash,
    RepetitionCode,
    StorySystemV2,
)
from embeddings.config.system_prompts import STORY_GENERATION
from watermarks import (
    GPT2Model,
    LanguageModel,
)
from attacks import (
    Attack,
    NGramShuffleAttack,
    ParaphraseAttack,
    SynonymAttack,
)
from attacks.translation import TranslationAttack

from ..utils import (
    BypassEncoder,
    CheckpointManager,
    CheckpointState,
    ExperimentConfig,
    ProgressTracker,
    TextLogger,
    index_reducer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def init_attacks(
    model: LanguageModel,
    client,
    attack_configs: list[dict] | None = None,
) -> dict[str, Attack]:
    if attack_configs is None:
        return {
            "n-gram": NGramShuffleAttack(model=model, n=3),
            "synonym": SynonymAttack(method="wordnet"),
            "paraphrase": ParaphraseAttack(
                client=client, model="gpt-4.1", temperature=0
            ),
            "translate": TranslationAttack(
                client=client, model="gpt-4.1", temperature=0
            ),
        }

    attacks = {}
    for cfg in attack_configs:
        if cfg["attack_type"] == "n-gram":
            attacks[cfg["label"]] = NGramShuffleAttack(model=model, n=3)
        elif cfg["attack_type"] == "synonym":
            attacks[cfg["label"]] = SynonymAttack(method="wordnet")
        elif cfg["attack_type"] == "paraphrase":
            attacks[cfg["label"]] = ParaphraseAttack(
                client=client,
                model="gpt-4.1",
                temperature=0.7,
            )
        elif cfg["attack_type"] == "translate":
            attacks[cfg["label"]] = TranslationAttack(
                client=client,
                model="gpt-4.1",
                temperature=0,
            )
        else:
            raise ValueError(f"Unknown attack type: {cfg['attack_type']}")

    return attacks


def apply_attack(
    attacks: dict[str, Attack],
    attack_key: str,
    stego_text: str,
    tampering: float,
    local: bool,
) -> str:
    print(f"======stego_text======\n{stego_text}")
    result = attacks[attack_key](stego_text, tampering, local)
    print(f"======attacked_text======\n{result}")
    return result


def compute_recovery_accuracy(original: list[int], recovered: list[int]) -> float:
    if len(original) != len(recovered):
        return 0.0
    correct = sum(o == r for o, r in zip(original, recovered))
    return correct / len(original)


def generate_random_messages(
    num_messages: int, num_bits: int, seed: int | None = None
) -> list[list[int]]:
    if seed is not None:
        random.seed(seed)
    return [
        [random.randint(0, 1) for _ in range(num_bits)] for _ in range(num_messages)
    ]


class StegoExperiment:
    """
    Runs steganography recovery experiments under various attacks.

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
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_path)
        self.progress = ProgressTracker()
        self.model = GPT2Model()
        self.attacks: dict[str, Attack] = {}
        self.messages = config.messages

        # Set up output paths
        self.output_dir = Path(config.output_path)
        self.summary_dir = self.output_dir / "summaries"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.text_logger = (
            TextLogger(self.output_dir / "texts_log.jsonl", config.max_saved_examples)
            if config.save_texts
            else None
        )

    def run(self) -> dict:
        # Load or initialize state
        if self.config.resume and self.checkpoint_mgr.exists():
            state = self.checkpoint_mgr.load()
            LOGGER.info(f"Resumed from checkpoint: {self.config.checkpoint_path}")
        else:
            state = CheckpointState()

        # Generate messages
        messages = (
            self.messages
            if self.messages is not None
            else generate_random_messages(
                self.config.num_messages, self.config.num_bits, self.config.seed
            )
        )
        print(f"messages: {messages}")

        # Save random state
        state.random_state = random.getstate()
        self.checkpoint_mgr.save(state)

        # Initialize attacks with configs to support local_mode
        self.attacks = init_attacks(
            self.model,
            self.config.system.client,
            self.config.attack_configs,
        )

        LOGGER.info("Starting experiment")

        # Process each message
        for msg_idx in range(state.message_index, self.config.num_messages):
            self._process_message(state, messages[msg_idx], msg_idx)

        # Final aggregation
        self._aggregate_final_results(state)

        LOGGER.info("Experiment finished; final results saved.")
        return state.all_ret

    def _process_message(
        self, state: CheckpointState, message: list[int], msg_idx: int
    ):
        """Process a single message through all attacks."""
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"Processing Message {msg_idx + 1}/{self.config.num_messages}")
        LOGGER.info(f"{'=' * 60}")

        # Generate stego texts if needed
        if not state.current_message_stego_complete:
            self._generate_stego_texts(state, message, msg_idx)
        else:
            LOGGER.info(
                f"Using {len(state.current_stego_texts)} pre-generated stego texts"
            )

        # Run attacks
        for a_idx in range(state.attack_index, len(self.config.attack_configs)):
            self._process_attack(state, message, msg_idx, a_idx)

        # Move to next message
        state.message_index = msg_idx + 1
        state.reset_for_new_message()
        self.checkpoint_mgr.save(state)

    def _generate_stego_texts(
        self, state: CheckpointState, message: list[int], msg_idx: int
    ):
        """Generate stego texts for a message."""
        LOGGER.info(f"Generating {self.config.num_stego_per_message} stego texts")

        bar = self.progress.create(
            "stego_gen",
            self.config.num_stego_per_message,
            f"Generating stego (msg {msg_idx})",
            initial=state.stego_gen_index,
        )

        for i in range(state.stego_gen_index, self.config.num_stego_per_message):
            stego_text = self.config.system.hide_message(message, self.config.history)
            state.current_stego_texts.append(stego_text)
            state.stego_gen_index = i + 1

            if self.config.checkpoint_after_each_stego:
                self.checkpoint_mgr.save(state)

            bar.update(1)

        self.progress.close("stego_gen")
        state.current_message_stego_complete = True
        self.checkpoint_mgr.save(state)
        LOGGER.info(f"Stego generation complete for message {msg_idx}")

    def _process_attack(
        self, state: CheckpointState, message: list[int], msg_idx: int, attack_idx: int
    ):
        """Process a single attack configuration."""
        attack_cfg = self.config.attack_configs[attack_idx]
        LOGGER.info(
            f"\nAttack {attack_idx + 1}/{len(self.config.attack_configs)}: "
            f"{attack_cfg['label']} (Message {msg_idx})"
        )

        # Initialize results structures
        if attack_cfg["label"] not in state.all_ret:
            state.all_ret[attack_cfg["label"]] = {
                "results_bitwise": [],
                "results_perfect": [],
                "data_lines_bitwise": ["Tampering_Percentage\tBitwise_Accuracy"],
                "data_lines_perfect": ["Tampering_Percentage\tPerfect_Recovery_Rate"],
                "tampering_level_data": {},
            }
        if attack_cfg["label"] not in state.message_results:
            state.message_results[attack_cfg["label"]] = {"tampering_level_data": {}}

        # Process tampering levels
        bar = self.progress.create(
            "tampering",
            len(self.config.tampering_levels),
            f"Tampering ({attack_cfg['label']}, msg {msg_idx})",
            initial=state.tampering_index,
        )

        for t_idx in range(state.tampering_index, len(self.config.tampering_levels)):
            self.progress.set_position("tampering", t_idx)
            self._process_tampering_level(state, message, msg_idx, attack_cfg, t_idx)

        self.progress.close("tampering")

        # Advance attack
        state.attack_index = attack_idx + 1
        state.reset_for_new_attack()
        self.checkpoint_mgr.save(state)

    def _process_tampering_level(
        self,
        state: CheckpointState,
        message: list[int],
        msg_idx: int,
        attack_cfg: dict,
        t_idx: int,
    ):
        """Process a single tampering level."""
        tp = self.config.tampering_levels[t_idx]
        msg_results = state.message_results[attack_cfg["label"]]

        if tp not in msg_results["tampering_level_data"]:
            msg_results["tampering_level_data"][tp] = {
                "perfect_scores": [],
                "bitwise_scores": [],
            }

        # Process stego texts
        bar = self.progress.create(
            "stego",
            len(state.current_stego_texts),
            f"Stego texts@tp={tp}",
            position=1,
            leave=False,
            initial=state.stego_index,
        )

        for s_idx in range(state.stego_index, len(state.current_stego_texts)):
            self._process_stego_text(state, message, msg_idx, attack_cfg, tp, s_idx)
            bar.update(1)

        self.progress.close("stego")

        # Aggregate tampering level results
        self._aggregate_tampering_results(state, attack_cfg["label"], tp)

        # Advance tampering
        state.tampering_index = t_idx + 1
        state.reset_for_new_tampering()
        self.checkpoint_mgr.save(state)

    def _process_stego_text(
        self,
        state: CheckpointState,
        message: list[int],
        msg_idx: int,
        attack_cfg: dict,
        tp: float,
        stego_idx: int,
    ):
        """Process a single stego text through multiple runs."""
        stego_text = state.current_stego_texts[stego_idx]
        success_count = 0
        bit_score = 0.0

        bar = self.progress.create(
            "runs",
            self.config.runs,
            f"Runs (stego {stego_idx})",
            position=2,
            leave=False,
            initial=state.run_index,
        )

        for run_i in range(state.run_index, self.config.runs):
            perfect, bitwise, recovered, attacked = self._single_run(
                message, stego_text, attack_cfg, tp
            )
            success_count += perfect
            bit_score += bitwise

            # Log if enabled
            if self.text_logger:
                self.text_logger.log(
                    attack_cfg["label"],
                    attack_cfg["attack_type"],
                    tp,
                    msg_idx,
                    stego_idx,
                    run_i,
                    message,
                    stego_text,
                    attacked,
                    recovered,
                )
                state.texts_saved_count = self.text_logger.count

            # Checkpoint after each run
            state.run_index = run_i + 1
            self.checkpoint_mgr.save(state)
            bar.update(1)

        self.progress.close("runs")

        # Store aggregated scores
        runs = self.config.runs
        msg_results = state.message_results[attack_cfg["label"]][
            "tampering_level_data"
        ][tp]
        msg_results["perfect_scores"].append(success_count / runs if runs else 0.0)
        msg_results["bitwise_scores"].append(bit_score / runs if runs else 0.0)

        state.stego_index = stego_idx + 1
        state.reset_for_new_stego()
        self.checkpoint_mgr.save(state)

    def _single_run(
        self, message: list[int], stego_text: str, attack_cfg: dict, tp: float
    ) -> tuple[int, float, list[int], str]:
        """Execute a single attack-and-recover run."""
        # Use attack label as key since we're now keying by label
        attacked = apply_attack(
            self.attacks, attack_cfg["label"], stego_text, tp, attack_cfg["local"]
        )

        recovered = self.config.system.recover_message(attacked)
        print(f"message: {message}, recovered: {recovered}")

        perfect = 1 if recovered == message else 0

        encoded_truth = self.config.system.encoder.encode(message)
        encoded_rec = self.config.system.encoder.encode(recovered)

        if encoded_truth is None or encoded_rec is None:
            bitwise = 0.0
        else:
            bitwise = compute_recovery_accuracy(encoded_truth, encoded_rec)

        return perfect, bitwise, recovered, attacked

    def _aggregate_tampering_results(
        self, state: CheckpointState, attack_label: str, tp: float
    ):
        """Aggregate results for a tampering level across stego texts."""
        tp_data = state.message_results[attack_label]["tampering_level_data"][tp]

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

        # Store in global results
        if tp not in state.all_ret[attack_label]["tampering_level_data"]:
            state.all_ret[attack_label]["tampering_level_data"][tp] = {
                "perfect_per_message": [],
                "bitwise_per_message": [],
            }
        state.all_ret[attack_label]["tampering_level_data"][tp][
            "perfect_per_message"
        ].append(perfect_avg)
        state.all_ret[attack_label]["tampering_level_data"][tp][
            "bitwise_per_message"
        ].append(bitwise_avg)

    def _aggregate_final_results(self, state: CheckpointState):
        """Compute final aggregated results and save summaries."""
        LOGGER.info("\nFinal aggregation across all messages")

        for attack_label, attack_data in state.all_ret.items():
            results_bitwise = []
            results_perfect = []
            data_lines_bitwise = ["Tampering_Percentage\tBitwise_Accuracy"]
            data_lines_perfect = ["Tampering_Percentage\tPerfect_Recovery_Rate"]

            for tp in self.config.tampering_levels:
                tp_data = attack_data["tampering_level_data"][tp]

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

            attack_data["results_bitwise"] = results_bitwise
            attack_data["results_perfect"] = results_perfect
            attack_data["data_lines_bitwise"] = data_lines_bitwise
            attack_data["data_lines_perfect"] = data_lines_perfect

            # Save tsv files
            fname = attack_label.replace(" ", "_")
            (self.summary_dir / f"{fname}_bitwise.tsv").write_text(
                "\n".join(data_lines_bitwise) + "\n"
            )
            (self.summary_dir / f"{fname}_perfect.tsv").write_text(
                "\n".join(data_lines_perfect) + "\n"
            )

        self.checkpoint_mgr.save(state)


def plot_recovery_results(
    tampering_levels: list[float],
    attack_labels: list[str],
    results: dict,
    output_path: str,
):
    """Generate and save recovery accuracy plots."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bitwise accuracy plot
    plt.figure(figsize=(8, 6))
    for label in attack_labels:
        plt.plot(tampering_levels, results[label]["results_bitwise"], label=label)

        # Save data file
        fname = label.lower().replace(" ", "_")
        (output_dir / f"{fname}_bitwise.txt").write_text(
            "\n".join(results[label]["data_lines_bitwise"])
        )

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Bitwise Recovery Accuracy")
    plt.title("Bitwise Recovery Accuracy Across Attacks")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "all_attacks_bitwise.png")
    plt.close()

    # Perfect recovery plot
    plt.figure(figsize=(8, 6))
    for label in attack_labels:
        plt.plot(tampering_levels, results[label]["results_perfect"], label=label)

        fname = label.lower().replace(" ", "_")
        (output_dir / f"{fname}_perfect.txt").write_text(
            "\n".join(results[label]["data_lines_perfect"])
        )

    plt.xlabel("Tampering Percentage")
    plt.ylabel("Perfect Recovery Rate")
    plt.title("Perfect Recovery Rate Across Attacks")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "all_attacks_perfect.png")
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    tp = [1.0]

    client = openai.OpenAI()
    local_client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1")

    ## Use a callback to manipulate the PCA hash
    hash_fn = PCAHash(
        pca_dir=pca_artifacts_dir("creative_stories"),
        bit_reducer=index_reducer(2),
    )

    # Error correction: controls stegotext length
    # - Repetition: repetition * num_bits
    # - Convolution: 4 * (num_bits + K - 1)
    ecc = RepetitionCode(1)
    # ecc = ConvolutionalCode(1, 3)

    system_prompt = STORY_GENERATION.format(
        items="Fedral agent Stanley Cooper, an espionage conducted by Slovenian Government, A novel nuclear weapon",
        boring_theme="An agent stopped an espionage.",
    )

    system = StorySystemV2(
        client=client,
        error_correction=ecc,
        local_client=local_client,
        local_model="Qwen3.5-9B-Q4_K_M.gguf",
        n_slots=20,
        encoder=BypassEncoder(),
    )

    premise = "Federal agent Stanley Cooper investigates an espionage operation conducted by the Slovenian government involving a novel nuclear weapon. The story follows Cooper as he tracks a suspect, discovers secret plans, confronts danger, and attempts to escape with critical intelligence."

    attack_configs = [
        # {"label": "Paraphrase (local)", "attack_type": "paraphrase", "local": True},
        {"label": "Paraphrase (global)", "attack_type": "paraphrase", "local": False},
    ]

    config = ExperimentConfig(
        tampering_levels=tp,
        attack_configs=attack_configs,
        system=system,
        num_bits=18,
        num_messages=3,
        num_stego_per_message=2,
        runs=5,
        history=premise,
        seed=1228,
        checkpoint_path=Path("checkpoints/test/exp_checkpoint.pkl"),
        output_path=Path("figures/test/embedding_recovery_test"),
        save_texts=True,
        max_saved_examples=200,
        resume=True,
        checkpoint_after_each_stego=True,
    )

    experiment = StegoExperiment(config)
    results = experiment.run()
    print(results)

    # attack_labels = [cfg.label for cfg in attack_configs]
    # plot_recovery_results(
    #     tp, attack_labels, results, "./figures/embedding_recovery_test/"
    # )


if __name__ == "__main__":
    main()
