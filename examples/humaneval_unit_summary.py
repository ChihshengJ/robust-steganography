"""
Run the UnitSummarySystem on the first N HumanEval tasks.

Example usage:
    python3 examples/humaneval_unit_summary.py --count 5 --message "hello"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, override

from datasets import Dataset, load_dataset
from openai import OpenAI

from embeddings import Encoder, PCAHash
from embeddings.core.error_correction import RepetitionCode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate steganographic unit tests for HumanEval tasks."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of HumanEval problems to process (default: 5).",
    )
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="Secret message to hide across the generated tests.",
    )
    parser.add_argument(
        "--high-priority",
        type=int,
        default=2,
        help="Number of high-priority behaviors/tests (H) to generate before encoding bits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/humaneval_tests"),
        help="Directory where generated test files will be written.",
    )
    parser.add_argument(
        "--hash-bits",
        type=int,
        default=1,
        help="Number of bits produced by the hash function (controls chunk size).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetition parameter for the ECC (must be >=1).",
    )
    parser.add_argument(
        "--bit-prefix",
        type=str,
        default="",
        help="Optional bitstring prefix (e.g. '101') prepended before encoding the secret.",
    )
    parser.add_argument(
        "--hash-seed",
        type=int,
        default=42,
        help="Random seed for the projection matrix (change to vary hash behavior).",
    )
    return parser.parse_args()


class BypassEncoder(Encoder):
    def __init__(self):
        pass

    @override
    def encode(self, data: list[int]) -> list[int]:
        return data

    @override
    def decode(self, bits: list[int]) -> list[int]:
        return bits


def prepare_system(
    client: OpenAI,
    high_priority: int,
    hash_bits: int,
    repetitions: int,
    bit_prefix: str,
    hash_seed: int,
) -> UnitSummarySystem:
    # hash_fn = RandomProjectionHash(num_bits=hash_bits, seed=hash_seed)
    hash_fn = PCAHash(
        pca_dir="src/pca/unit_test/artifacts", model_length=6, start=2, end=3
    )
    # currently the PCA is trained for maximum 5-bits hash

    ecc = RepetitionCode(repetitions=repetitions)
    prefix_bits: list[int] = []
    cleaned = bit_prefix.strip()
    if cleaned:
        for char in cleaned:
            if char not in ("0", "1"):
                raise ValueError("--bit-prefix must only contain 0/1 characters")
            prefix_bits.append(int(char))

    return UnitSummarySystem(
        client=client,
        key=high_priority,
        hash_function=hash_fn,
        error_correction=ecc,
        encoder=BypassEncoder(),
        bit_prefix=prefix_bits,
        sampled_hash=True,
    )


def iter_humaneval_prompts(limit: int) -> Iterable[dict]:
    dataset = load_dataset("openai/openai_humaneval", split="test")
    assert isinstance(dataset, Dataset)
    take = min(limit, len(dataset))
    for example in dataset.select(range(take)):
        yield example


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    system = prepare_system(
        client=client,
        high_priority=args.high_priority,
        hash_bits=args.hash_bits,
        repetitions=args.repetitions,
        bit_prefix=args.bit_prefix,
        hash_seed=args.hash_seed,
    )

    for example in iter_humaneval_prompts(args.count):
        task_id = example["task_id"]
        problem_statement = example["prompt"]
        output_path = args.output_dir / f"{task_id}_tests.py"
        print(f"Encoding task {task_id} -> {output_path}")
        stego = system.hide_message(
            data=[0, 1, 1],
            problem=problem_statement,
        )
        recovered = system.recover_message(stego)

        # output_path.write_text(test_suite, encoding="utf-8")
        print(f"data: {[0, 1, 1]}, recovered: {recovered}")


if __name__ == "__main__":
    main()
