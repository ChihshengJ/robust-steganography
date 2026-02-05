import re
from typing import Any

import numpy as np
from nltk.tokenize import sent_tokenize

from ..config.constants import BacktrackConfig
from ..config.system_prompts import (
    FACT_CONTINUATION_ANCHORED,
    FACT_DECOMPOSE,
    FACT_GENERATION_ANCHORED,
    FACT_SUMMARY_STRICT,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from ..utils.sample_utils import (
    BacktrackingEncoder,
    RejectionSampler,
    StepChoice,
    check_near_duplicate
)
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction
from .steg_system import StegSystem


def build_prompt_for_facts(history: dict, system_prompt: str, covered: set[str]) -> tuple[str, str]:
    def extract_topics(facts: list[str]) -> str:
        keywords = {
            word.strip(".,;:\"'-()[]")
            for fact in facts
            for word in fact.split()
            if word and word[0].isupper() and len(word) > 3
        }
        return ", ".join(sorted(keywords)[:8]) if keywords else "main event details"

    all_covered = history["base_facts"] + history["optional_facts"]
    facts_numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(all_covered))
    topic_hints = extract_topics(all_covered)
    prompt =  f"""
{history["article"]}
FACTS ALREADY EXTRACTED ({len(all_covered)} total) - you must find something DIFFERENT than these:
{facts_numbered}

TOPICS ALREADY COVERED: {topic_hints}

Extract ONE NEW fact that:
- Covers an event that occurred LATER IN TIME than all previously extracted facts on the list
- Is a single complete sentence
- Contains specific names, numbers, or dates from the article

Look for: secondary details, specific quotes, background context, reactions, locations, times.

Continue with fact {len(all_covered) + 1}:
"""
    system_prompt_mod = system_prompt
    if len(covered) > 0:
        system_prompt_mod += f"\nAdditionally, avoid these facts that are already covered: \n- {'\n- '.join(list(covered)[-7:])}"

    return system_prompt_mod, prompt


def update_fact_history(initial_history: dict, steps: list[StepChoice]) -> dict:
    return {
        "article": initial_history["article"],
        "base_facts": initial_history["base_facts"],
        "optional_facts": [choice.message for choice in steps],
    }


def clean_fact_response(response: str, history: dict) -> str:
    response = response.strip()
    expected_num = len(history["base_facts"]) + len(history["optional_facts"]) + 1

    patterns = [
        rf"^{expected_num}[.):\s]+",
        r"^\d+[.):\s]+",
        r"^[Ff]act[:\s]+",
        r"^[Nn]ew [Ff]act[:\s]+",
        r"^[-•*]\s*",
    ]
    for pattern in patterns:
        response = re.sub(pattern, "", response).strip()

    return response.strip("\"'")


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
            raise ValueError("key must be non-negative")
        super().__init__(client, hash_function, error_correction, encoder)
        self.message_length: int | None = None
        self.key = key
        self.hash_output_length = getattr(hash_function, "output_length")
        self.backtrack_config = backtrack_config or BacktrackConfig()

        sampler = RejectionSampler(
            prompt_builder=build_prompt_for_facts,
            duplicate_checker=check_near_duplicate,
            response_cleaner=clean_fact_response,
            count_duplicates=False,
        )
        self._encoder = BacktrackingEncoder(
            sampler=sampler,
            config=self.backtrack_config,
            history_updater=update_fact_history,
        )

        self._base_facts: list[str] = []
        self._optional_facts: list[str] = []

    def encode(
        self,
        chunks: list[list[int]],
        history: str,
        system_prompt: str,
        max_length: int = 200,
        base_facts: list[str] | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[list[str], list]:
        initial_history = {
            "article": history,
            "base_facts": base_facts or [],
            "optional_facts": [],
        }
        return self._encoder.encode(
            client=self.client,
            chunks=[np.array(lst) for lst in chunks],
            initial_history=initial_history,
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
        )

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        chunks, self.message_length = self._encode_to_chunks(data)

        ### Generate base facts
        base_facts_raw = generate_response(
            prompt=seed,
            system_prompt=FACT_GENERATION_ANCHORED.format(k=self.key),
            max_length=1500,
            temperature=0,
        )
        self._base_facts = self._parse_numbered_facts(base_facts_raw)

        print(f"\n{'=' * 60}")
        print(f"Base facts ({len(self._base_facts)}):")
        print(f"{'=' * 60}")
        for f in self._base_facts:
            print(f"  {f}")

        print(f"\n{'=' * 60}")
        print(f"Generating {len(chunks)} optional facts...")
        print(f"{'=' * 60}")

        ### Generate optional facts
        self._optional_facts, _ = self.encode(
            chunks=chunks,
            history=seed,
            base_facts=self._base_facts,
            system_prompt=FACT_CONTINUATION_ANCHORED,
            max_length=200,
            temperature=1.0,
        )

        print(f"\n{'=' * 60}")
        print(f"Optional facts ({len(self._optional_facts)}):")
        print(f"{'=' * 60}")
        for i, f in enumerate(self._optional_facts):
            print(f"  {self.key + i + 1}. {f}")

        ### Generate summary
        all_facts = self._base_facts + self._optional_facts
        facts_formatted = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(all_facts))

        print(f"\n{'=' * 60}")
        print("Generating summary...")
        print(f"{'=' * 60}")

        stego_text = generate_response(
            prompt=facts_formatted,
            system_prompt=FACT_SUMMARY_STRICT,
            max_length=3000,
            temperature=0,
        ).strip()

        print(f"\nGenerated summary ({len(sent_tokenize(stego_text))} sentences):")
        print(f"  {stego_text[:300]}...")

        ### Check summary text alignment
        self._verify_alignment(stego_text, all_facts)
        return stego_text

    def recover_message(self, stego_text: str):
        if self.message_length is None:
            raise ValueError("No message length set. Run hide_message first or set message_length.")

        expected_optional = self.message_length // self.hash_output_length
        expected_total = self.key + expected_optional

        print(f"\n{'=' * 60}")
        print("Recovery - LLM Decomposition")
        print(f"{'=' * 60}")
        print(f"  Expected facts: {expected_total} (key={self.key}, optional={expected_optional})")

        all_facts = self._decompose_summary(stego_text, expected_total)
        print(f"  Extracted facts: {len(all_facts)}")

        if len(all_facts) != expected_total:
            print(f"    WARNING: Expected {expected_total} facts, got {len(all_facts)}")

        optional_facts = all_facts[self.key : self.key + expected_optional]
        print(f"\n  Optional facts for decoding ({len(optional_facts)}):")
        for i, f in enumerate(optional_facts):
            print(f"    {self.key + i + 1}. {f}")

        if len(optional_facts) < expected_optional:
            print(f"    Only {len(optional_facts)} optional facts, expected {expected_optional}")

        embeddings = get_embeddings_in_batch(self.client, optional_facts)
        return self._decode_from_embeddings(embeddings, self.message_length)

    def _decompose_summary(self, summary: str, num_facts: int) -> list[str]:
        response = generate_response(
            prompt=summary,
            system_prompt=FACT_DECOMPOSE.format(num_facts=num_facts),
            max_length=3000,
            temperature=0,
        )
        facts = []
        for part in response.split("[sep]"):
            fact = part.strip()
            if not fact:
                continue
            fact = re.sub(r"^\d+[.):\s]+", "", fact).strip()
            fact = re.sub(r"^[-•*]\s*", "", fact).strip()
            if fact:
                facts.append(fact)
        return facts

    def _parse_numbered_facts(self, raw_text: str) -> list[str]:
        facts = []
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for i in range(1, 100):
                prefix = f"{i}."
                if line.startswith(prefix):
                    fact_text = line[len(prefix):].strip()
                    if fact_text:
                        facts.append(fact_text)
                    break
        return facts

    def _verify_alignment(self, summary: str, facts: list[str]) -> None:
        sentences = sent_tokenize(summary)
        print("\n  Alignment check:")
        if len(sentences) != len(facts):
            print(f"    MISMATCH: {len(sentences)} sentences vs {len(facts)} facts")
        else:
            print(f"    OK: {len(sentences)} sentences = {len(facts)} facts")
