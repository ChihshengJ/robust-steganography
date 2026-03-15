import json
import re
from typing import Any

import numpy as np

from ..config.constants import BacktrackConfig
from ..config.cot_prompts import (
    BASE_GENERATION,
    DECOMPOSE,
    STEP_CONTINUATION,
    STEP_CONTINUATION_TEMPLATE,
    SYNTHESIZE,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from ..utils.sample_utils import (
    BacktrackingEncoder,
    RejectionSampler,
    StepChoice,
    check_near_duplicate,
)
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction
from .steg_system import StegSystem


def build_prompt(
    history: dict, system_prompt: str, covered: set[str]
) -> tuple[str, str]:
    base_steps = history["base_steps"]
    optional_steps = history["optional_steps"]
    all_steps = base_steps + optional_steps

    prev_lines = "\n".join(f"  Step {i + 1}: {s}" for i, s in enumerate(all_steps))

    prompt = STEP_CONTINUATION_TEMPLATE.format(
        question=history["question"],
        previous_steps=prev_lines or "(none yet)",
        n_previous=len(all_steps),
    )

    sys = system_prompt
    if covered:
        recent = list(covered)[-5:]
        sys += (
            "\n\nPrevious attempts for this step (write something DIFFERENT):\n"
            + "\n".join(f"- {s[:120]}" for s in recent)
        )
    return sys, prompt


def update_history(initial_history: dict, steps: list[StepChoice]) -> dict:
    return {
        **initial_history,
        "optional_steps": [step.message for step in steps],
    }


def clean_response(response: str, history: dict) -> str:
    response = response.strip()
    response = re.sub(r"^\d+[.):\s]+", "", response).strip()
    response = re.sub(r"^[-\u2022*]\s*", "", response).strip()
    response = re.sub(r"^(?:Step\s+\d+\s*[:.\-]\s*)", "", response, flags=re.IGNORECASE).strip()
    return response.strip("\"'")


class CotSystem(StegSystem):
    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        backtrack_config: BacktrackConfig | None = None,
    ):
        if key < 0:
            raise ValueError("key must be non-negative")
        super().__init__(client, hash_function, error_correction, encoder)
        self.key = key
        self.message_length: int | None = None
        self.hash_output_length = getattr(hash_function, "output_length")
        self.backtrack_config = backtrack_config or BacktrackConfig()

        sampler = RejectionSampler(
            prompt_builder=build_prompt,
            duplicate_checker=check_near_duplicate,
            response_cleaner=clean_response,
            count_duplicates=False,
        )
        self._encoder = BacktrackingEncoder(
            sampler=sampler,
            config=self.backtrack_config,
            history_updater=update_history,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: dict,
        system_prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[list[str], list]:
        return self._encoder.encode(
            client=self.client,
            chunks=[np.array(c) for c in chunks],
            initial_history=history,
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
        )

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        question = seed
        chunks, self.message_length = self._encode_to_chunks(data)
        expected_optional = len(chunks)

        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print(f"Encoding {expected_optional} chunks, {self.key} base steps")
        print(f"{'=' * 60}")

        base_steps = self._generate_base_steps(question) if self.key > 0 else []

        print(f"\nBase steps ({len(base_steps)}):")
        for i, s in enumerate(base_steps):
            print(f"  Step {i + 1}: {s}")

        print(f"\nGenerating {expected_optional} encoding steps...")

        initial_history = {
            "question": question,
            "base_steps": base_steps,
            "optional_steps": [],
        }

        optional_steps, _ = self.encode(
            chunks=chunks,
            history=initial_history,
            system_prompt=STEP_CONTINUATION,
            max_length=200,
            temperature=1,
        )

        print(f"\nEncoding steps ({len(optional_steps)}):")
        for i, s in enumerate(optional_steps):
            print(f"  Step {self.key + i + 1}: {s}")

        all_steps = base_steps + optional_steps
        # stego_text = self._synthesize(all_steps, question)
        stego_text = " ".join(all_steps)
        return stego_text

    def recover_message(self, stego_text: str, **kwargs) -> Any:
        if self.message_length is None:
            raise ValueError(
                "No message length set. Run hide_message first or set message_length."
            )

        expected_optional = self.message_length // self.hash_output_length
        expected_total = self.key + expected_optional

        print(f"\n{'=' * 60}")
        print("Recovery - LLM Decomposition")
        print(f"{'=' * 60}")
        print(f"  Expected: {expected_total} steps (base={self.key}, encoding={expected_optional})")

        claims = self._decompose(stego_text, expected_total)
        print(f"  Extracted: {len(claims)} claims")

        for i, s in enumerate(claims):
            tag = "base" if i < self.key else "enc"
            print(f"    [{tag}] {i + 1}. {s[:120]}...")

        if len(claims) != expected_total:
            print(f"  WARNING: Expected {expected_total} claims, got {len(claims)}")

        optional_claims = claims[self.key : self.key + expected_optional]

        if len(optional_claims) < expected_optional:
            print(f"  WARNING: Only {len(optional_claims)} encoding claims, expected {expected_optional}")

        embeddings = get_embeddings_in_batch(self.client, optional_claims)
        return self._decode_from_embeddings(embeddings, self.message_length)

    def _generate_base_steps(self, question: str) -> list[str]:
        prompt = f"Question: {question}"

        response = generate_response(
            prompt=prompt,
            system_prompt=BASE_GENERATION.format(k=self.key),
            max_length=2000,
            temperature=0,
        )

        steps = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[.):\s]+", "", line).strip()
            line = re.sub(r"^(?:Step\s+\d+\s*[:.\-]\s*)", "", line, flags=re.IGNORECASE).strip()
            if line:
                steps.append(line)

        if len(steps) < self.key:
            print(f"  WARNING: Generated {len(steps)} base steps, expected {self.key}")

        return steps[: self.key]

    def _synthesize(self, steps: list[str], question: str) -> str:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        response = generate_response(
            prompt=numbered,
            system_prompt=SYNTHESIZE.format(question=question),
            max_length=4000,
            temperature=0,
        )
        return response

    def _decompose(self, text: str, expected_total: int) -> list[str]:
        response = generate_response(
            prompt=text,
            system_prompt=DECOMPOSE.format(n=expected_total),
            max_length=4000,
            temperature=0,
        )

        claims = []
        for part in response.split("[sep]"):
            claim = part.strip()
            if not claim:
                continue
            claim = re.sub(r"^\d+[.):\s]+", "", claim).strip()
            claim = re.sub(r"^[-\u2022*]\s*", "", claim).strip()
            if claim:
                claims.append(claim)

        if len(claims) != expected_total:
            print(f"  Decomposition returned {len(claims)} claims, expected {expected_total}")

        return claims
