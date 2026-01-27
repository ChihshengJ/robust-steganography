from typing import Any

import numpy as np

from embeddings import StegSystem

from ..config.constants import BacktrackConfig
from ..config.system_prompts import (
    FACT_CONTINUATION,
    FACT_DECOMPOSE,
    FACT_GENERATION,
    FACT_SUMMARY,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from ..utils.sample_utils import BacktrackingEncoder, RejectionSampler
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction


class SummarySystem(StegSystem):
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
            raise ValueError(
                "key must be non-negative (represents high priority behaviors)"
            )
        super().__init__(client, hash_function, error_correction, encoder)
        self.error_encoded_length: int | None = None
        self.key = key
        self.hash_output_length = getattr(hash_function, "output_length")
        self.backtrack_config = backtrack_config or BacktrackConfig()
        self._backtracking_encoder = BacktrackingEncoder(
            sampler=RejectionSampler(),
            config=self.backtrack_config,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[list[str], list]:
        facts, embeddings = self._backtracking_encoder.encode(
            client=self.client,
            chunks=[np.array(lst) for lst in chunks],
            initial_history=history if isinstance(history, list) else [history],
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
            use_prohibitions=False,
        )
        return facts, embeddings

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        """
        Given an article, generate K key facts listed in descending order of importance.
        Then ask the LLM to generate the next fact and hash it efficiently, avoid duplicate facts by maintaining a list of known facts.
        Keep generating until m bits are reached.
        Last write a summary based on the K + m facts.
        """

        chunks, self.message_length = self._encode_to_chunks(data)

        # base facts generation
        base_facts = generate_response(
            client=self.client,
            prompt=seed,
            system_prompt=FACT_GENERATION.format(k=self.key),
            max_length=1000,
            temperature=0,
        )

        print(f"base_facts: {base_facts}")

        # optional facts
        prompt_for_optional_facts: str = (
            seed + "\nKey facts:\n" + base_facts + "\nOptional facts:\n"
        )
        optional_facts, _ = self.encode(
            chunks=chunks,
            history=prompt_for_optional_facts,
            system_prompt=FACT_CONTINUATION,
            max_length=1000,
            temperature=1,
        )

        print(f"optional facts: {optional_facts}")

        # generate summary based on all facts
        prompt_for_summary = [base_facts + "\n".join(optional_facts) + "\nSummary:\n"]

        print(f"prompt for summary: {prompt_for_summary}")

        stego_text = generate_response(
            client=self.client,
            prompt=prompt_for_summary,
            system_prompt=FACT_SUMMARY,
            max_length=1000,
            temperature=0,
        ).strip()

        print(f"cover_text: {stego_text}")

        return stego_text

    def recover_message(self, stego_text: str):
        """
        Decompose article into K+m facts,
        """
        optional_facts = self._decompose_summary(stego_text)[self.key :]
        print(
            f"optional facts after decomposed: {len(optional_facts)},\n {optional_facts}"
        )
        embeddings = get_embeddings_in_batch(self.client, optional_facts)
        return self._decode_from_embeddings(embeddings, self.message_length)

    def _decompose_summary(self, stego_text: str):
        response = generate_response(
            client=self.client,
            prompt=stego_text,
            system_prompt=FACT_DECOMPOSE.format(num=self.key + self.message_length),
            max_length=1000,
            temperature=0,
        )
        print(f"decompse response: {response}")
        facts = [f.strip() for f in response.split(sep="[sep]") if f.strip()]
        print(f"facts decomposed: {facts}")
        return facts
