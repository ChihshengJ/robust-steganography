import hashlib
import json
import math
import random
import time
from typing import Any

from systems.core.hash_functions import BitsPerGroupStub

from .encoder import Encoder
from .error_correction import ErrorCorrection
from .steg_system import StegSystem
from ..config.topicqa_prompts import SUBTOPIC_PROMPT, ENCODE_PROMPT, DECODE_PROMPT
from ..utils.new_text import llm


def _parse_topic_list(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _bits_to_index(bits: list[int]) -> int:
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def _index_to_bits(idx: int, n_bits: int) -> list[int]:
    return [(idx >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


class TopicQASystem(StegSystem):
    def __init__(
        self,
        client: Any,
        error_correction: ErrorCorrection,
        local_client: Any,
        local_model: str,
        n_subtopics: int = 12,
        group_size: int = 2,
        response_model: str = "gpt-4.1",
        decoder_model: str = "gpt-4.1",
        key: str = "default",
        encoder: Encoder | None = None,
        response_temperature: float = 0.7,
    ) -> None:
        if group_size < 2 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a power of 2, got {group_size}")

        bits_per_group = int(math.log2(group_size))
        stub = BitsPerGroupStub(bits_per_group)

        super().__init__(client, stub, error_correction, encoder)

        self.local_client = local_client
        self.local_model = local_model
        self.n_subtopics = n_subtopics
        self.group_size = group_size
        self.bits_per_group = bits_per_group
        self.response_model = response_model
        self.decoder_model = decoder_model
        self.key = key
        self.response_temperature = response_temperature

        self._question: str | None = None
        self._last_metadata: dict | None = None

    @property
    def n_groups(self) -> int:
        return self.n_subtopics // self.group_size

    @property
    def raw_capacity_bits(self) -> int:
        return self.n_groups * self.bits_per_group

    @property
    def question(self) -> str | None:
        return self._question

    @question.setter
    def question(self, value: str | None) -> None:
        self._question = value

    def generate_subtopics(self, question: str) -> list[str]:
        raw = llm(
            self.local_client,
            self.local_model,
            SUBTOPIC_PROMPT.format(n=self.n_subtopics, question=question),
            temperature=0,
            max_tokens=2500,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        topics = _parse_topic_list(raw)
        if len(topics) > self.n_subtopics:
            topics = topics[: self.n_subtopics]
        if len(topics) < self.n_subtopics:
            print(
                f"  WARNING: local model returned {len(topics)} topics, expected {self.n_subtopics}"
            )
        return topics

    def group_subtopics(self, topics: list[str]) -> list[list[str]]:
        rng = random.Random(hashlib.sha256(self.key.encode()).hexdigest())
        indices = list(range(len(topics)))
        rng.shuffle(indices)
        groups = []
        for i in range(0, len(indices) - self.group_size + 1, self.group_size):
            groups.append([topics[indices[j]] for j in range(i, i + self.group_size)])
        return groups

    def _select_subtopics(
        self, groups: list[list[str]], chunks: list[list[int]]
    ) -> list[str]:
        selected = []
        for group, chunk in zip(groups, chunks):
            idx = _bits_to_index(chunk)
            selected.append(group[idx % len(group)])
        return selected

    def _generate_response(
        self, question: str, selected_topics: list[str], forbidden_topics: list[str]
    ) -> str:
        topics_str = "\n".join(f"- {t}" for t in selected_topics)
        forbidden_str = "\n".join(f"- {t}" for t in forbidden_topics)
        prompt = ENCODE_PROMPT.format(
            topics_str=topics_str, forbidden_str=forbidden_str, question=question
        )
        return llm(
            self.client,
            self.response_model,
            prompt,
            temperature=self.response_temperature,
            max_tokens=2500,
        )

    def _decode_group(self, response: str, question: str, group: list[str]) -> int:
        options = "\n".join(f"({chr(65 + i)}) {t}" for i, t in enumerate(group))
        letters = ", ".join(chr(65 + i) for i in range(len(group)))
        prompt = DECODE_PROMPT.format(
            question=question,
            response=response,
            options=options,
            letters=letters,
        )
        answer = (
            llm(
                self.client,
                self.decoder_model,
                prompt,
                temperature=0,
            )
            .strip()
            .upper()
        )
        for i in range(len(group)):
            if chr(65 + i) in answer:
                return i
        return 0

    def paraphrase(self, text: str, model: str | None = None) -> str:
        return llm(
            self.client,
            model or self.response_model,
            f"Rewrite this text completely in your own words, "
            f"preserving all informational content:\n\n{text}",
            temperature=0.7,
            max_tokens=2000,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str = "",
        max_length: int = 200,
        **kwargs,
    ) -> tuple[str, dict]:
        question = history if isinstance(history, str) else history[0]
        topics = self.generate_subtopics(question)
        groups = self.group_subtopics(topics)

        if len(chunks) > len(groups):
            raise ValueError(
                f"Payload requires {len(chunks)} groups but only {len(groups)} available "
                f"({self.n_subtopics} subtopics / {self.group_size} per group)"
            )

        selected = self._select_subtopics(groups, chunks)
        forbidden = [
            t for group, sel in zip(groups, selected) for t in group if t != sel
        ]
        response = self._generate_response(question, selected, forbidden)

        metadata = {
            "topics": topics,
            "groups": groups,
            "selected": selected,
            "chunks": chunks,
        }
        return response, metadata

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        self._question = seed
        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        print(
            f"Payload: {len(chunks)} groups × {self.bits_per_group} bits = "
            f"{len(chunks) * self.bits_per_group} channel bits "
            f"(capacity: {self.raw_capacity_bits})"
        )

        response, metadata = self.encode(chunks, seed)
        self._last_metadata = metadata
        print(f"Selected subtopics: {metadata['selected']}")
        return response

    def recover_message(self, stego_text: str) -> Any:
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )
        if self._question is None:
            raise ValueError("No question set. Run hide_message first or set question.")

        expected_chunks = -(-self._error_encoded_length // self.bits_per_group)

        topics = self.generate_subtopics(self._question)
        groups = self.group_subtopics(topics)

        if expected_chunks > len(groups):
            raise ValueError(
                f"Expected {expected_chunks} groups but only {len(groups)} available"
            )

        recovered_bits: list[list[int]] = []
        for gi in range(expected_chunks):
            detected_idx = self._decode_group(stego_text, self._question, groups[gi])
            bits = _index_to_bits(detected_idx, self.bits_per_group)
            recovered_bits.append(bits)
            group_labels = " | ".join(t[:30] for t in groups[gi])
            print(f"  group {gi}: detected={detected_idx} bits={bits} [{group_labels}]")

        m_bits = self.ecc.decode(recovered_bits, self._error_encoded_length)
        return self.encoder.decode(m_bits)
