import re
from typing import Any, List

import numpy as np

from ..config.system_prompts import (
    FACT_CONTINUATION,
    FACT_DECOMPOSE,
    FACT_GENERATION,
    FACT_SUMMARY,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction


class SummarySystem:
    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
    ):
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.error_encoded_length = None
        self.key = key

        # Get hash output length
        self.hash_output_length = getattr(hash_function, "output_length")

    def hide_message(self, data: Any, article: str) -> str:
        """
        Given an article, generate K key facts listed in descending order of importance.
        Then ask the LLM to generate the next fact and hash it efficiently, avoid duplicate facts by maintaining a list of known facts.
        Keep generating until m bits are reached.
        Last write a summary based on the K + m facts.
        """

        # encoding and ECC
        m_bits: list[int] = self.encoder.encode(data)
        m_encoded: List[int] = self.ecc.encode(m_bits)
        self.error_encoded_length = len(m_encoded)
        m_chunks = [
            m_encoded[i : i + self.hash_output_length]
            for i in range(0, len(m_encoded), self.hash_output_length)
        ]
        m_chunks = [
            nested + [0] * (self.hash_output_length - len(nested))
            for nested in m_chunks
        ]
        self.message_length = len(m_chunks)

        print(f"raw code being encoded: {m_chunks}")

        # base facts generation
        base_facts = generate_response(
            client=self.client,
            conversation_history=article,
            system_prompt=FACT_GENERATION.format(k=self.key),
            max_length=1000,
            temperature=0,
        )

        print(f"base_facts: {base_facts}")

        # optional facts
        prompt_for_optional_facts: str = (
            article + "\nKey facts:\n" + base_facts + "\nOptional facts:\n"
        )

        print(f"prompt for optional facts: {prompt_for_optional_facts}")
        optional_facts = self._encode(
            chunks=m_chunks,
            history=prompt_for_optional_facts,
        )

        print(f"optional facts: {optional_facts}")

        # generate summary based on all facts
        prompt_for_summary = (base_facts + "\n".join(optional_facts) + "\nSummary:\n",)

        print(f"prompt for summary: {prompt_for_summary}")

        cover_text = generate_response(
            client=self.client,
            conversation_history=prompt_for_summary,
            system_prompt=FACT_SUMMARY,
            max_length=1000,
            temperature=0,
        ).strip()

        print(f"cover_text: {cover_text}")

        return cover_text

    def recover_message(self, stego_text: str):
        """
        Decompose article into K+m facts,
        """
        optional_facts = self._decompose_summary(stego_text)[self.key :]
        print(
            f"optional facts after decomposed: {len(optional_facts)},\n {optional_facts}"
        )
        embeddings = get_embeddings_in_batch(self.client, optional_facts)
        bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        m_bits = self.ecc.decode(bits_encoded, self.error_encoded_length)
        return self.encoder.decode(m_bits)

    def _encode(self, chunks: list[list[int]], history: str) -> list[str]:
        facts = []
        prohibited = set()
        sampled_bits = np.nan
        for chunk in chunks:
            while not np.array_equal(sampled_bits, chunk):
                system_prompt = FACT_CONTINUATION.format(prohibited_facts=prohibited)
                # print(f"system prompt for optional facts: {system_prompt}")
                response = generate_response(
                    client=self.client,
                    conversation_history=history,
                    system_prompt=system_prompt,
                    max_length=1000,
                    temperature=0.7,
                )
                processed_response = re.sub(r"\[\d+\]", "", response)
                print(f"optional fact: {processed_response}")
                embedding = (
                    self.client.embeddings.create(
                        input=[processed_response], model="text-embedding-3-large"
                    )
                    .data[0]
                    .embedding
                )
                sampled_bits = self.hash_fn(np.array(embedding).reshape(1, -1))
                print(f"sampled bits: {sampled_bits}, desired_bits: {chunk}")
                if np.array_equal(sampled_bits, chunk):
                    history += response + "\n"
                    facts.append(response)
                    print(f"updated history: {history}")
                    sampled_bits = np.nan
                    prohibited.clear()
                    break
                else:
                    if processed_response not in prohibited:
                        prohibited.add(processed_response)
                    print(f"updated prohibited_facts: {prohibited}")
        return facts

    def _decompose_summary(self, stego_text: str):
        response = generate_response(
            client=self.client,
            conversation_history=stego_text,
            system_prompt=FACT_DECOMPOSE.format(num=self.key + self.message_length),
            max_length=1000,
            temperature=0,
        )
        print(f"decompse response: {response}")
        facts = [f.strip() for f in response.split(sep="[sep]") if f.strip()]
        print(f"facts decomposed: {facts}")
        return facts
