import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from ..config.system_prompts import (
    UNIT_BEHAVIOR_PLAN,
    UNIT_TEST_GENERATION,
    UNIT_TEST_SORT,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction


@dataclass
class BehaviorPlan:
    high: List[str]
    medium: List[str]
    low: List[str]

    @staticmethod
    def _parse_payload(payload: str) -> dict:
        """
        Try to parse raw model output into JSON.

        The model occasionally wraps the JSON with extra narration; attempt to
        trim to the outermost braces if needed before failing.
        """
        text = (payload or "").strip()
        if not text:
            raise ValueError("Model returned empty behavior JSON payload")

        candidates = [text]

        if not text.startswith("{") or not text.endswith("}"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError("Model returned invalid behavior JSON")

    @classmethod
    def from_json(cls, payload: str) -> "BehaviorPlan":
        data = cls._parse_payload(payload)

        def _clean(items: Iterable[str]) -> List[str]:
            return [
                item.strip() for item in items if isinstance(item, str) and item.strip()
            ]

        return cls(
            high=_clean(data.get("high", [])),
            medium=_clean(data.get("medium", [])),
            low=_clean(data.get("low", [])),
        )

    def payload_behaviors(self) -> List[Tuple[str, str]]:
        """Return non-high behaviors in global priority order."""
        payload = [("Medium", behavior) for behavior in self.medium]
        payload.extend(("Low", behavior) for behavior in self.low)
        return payload


class UnitSummarySystem:
    """
    encode/decode covert bits inside HumanEval-style test suites.

    H (self.key) designates the number of high-priority tests that are emitted before any
    steganographic encoding occurs. The remaining tests each encode one chunk of bits.
    """

    MAX_BEHAVIORS_PER_REQUEST = 50
    PLAN_RETRY_LIMIT = 4

    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        bit_prefix: Sequence[int] | None = None,
    ):
        if key < 0:
            raise ValueError(
                "key must be non-negative (represents high priority behaviors)"
            )
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.error_encoded_length: int | None = None
        self.hash_output_length = getattr(hash_function, "output_length")
        self.key = key
        self.message_length: int | None = None
        self.bit_prefix = (
            [int(bit) for bit in bit_prefix] if bit_prefix is not None else []
        )
        for bit in self.bit_prefix:
            if bit not in (0, 1):
                raise ValueError("bit_prefix must only contain 0 or 1 values.")

    def hide_message(self, data: Any, article: str) -> str:
        """
        Given a HumanEval problem statement (`article`), generate:
          1. H high-priority tests that anchor the suite behaviorally.
          2. Additional tests whose embeddings hash to the encoded message bits.
        """
        prefix = self.bit_prefix if self.bit_prefix else []
        m_bits = prefix + self.encoder.encode(data)
        encoded_bits: List[int] = self.ecc.encode(m_bits)
        self.error_encoded_length = len(encoded_bits)
        chunks = [
            encoded_bits[i : i + self.hash_output_length]
            for i in range(0, len(encoded_bits), self.hash_output_length)
        ]
        chunks = [
            chunk + [0] * (self.hash_output_length - len(chunk)) for chunk in chunks
        ]
        self.message_length = len(chunks)

        total_behaviors = self.key + self.message_length
        print(f"total behaviors: \n{total_behaviors}")
        print(f"article: \n{article}")
        behavior_plan = self._plan_behaviors(
            problem=article, total_behaviors=total_behaviors
        )
        print(f"initial behavior plan: \n{behavior_plan}")

        tests: List[str] = []
        if len(behavior_plan.high) < self.key:
            raise ValueError(
                f"Planner only produced {len(behavior_plan.high)} high priority behaviors, "
                f"but {self.key} were requested."
            )

        # Step 2: deterministic high-priority tests.
        for behavior in behavior_plan.high[: self.key]:
            test_code = self._generate_test(
                problem=article,
                behavior=behavior,
                priority="High",
                existing_tests=tests,
                target_bits=None,
                chunk_index=None,
            )
            tests.append(test_code)

        print(f"tests: \n{tests}")

        # Step 3+: steganographic payload.
        payload_behaviors = behavior_plan.payload_behaviors()
        if len(payload_behaviors) < self.message_length:
            raise ValueError(
                "Planner did not provide enough medium/low behaviors to encode the payload."
            )

        for idx, (chunk, (priority, behavior)) in enumerate(
            zip(chunks, payload_behaviors)
        ):
            target_bits = np.array(chunk)
            test_code = self._generate_test(
                problem=article,
                behavior=behavior,
                priority=priority,
                existing_tests=tests,
                target_bits=target_bits,
                chunk_index=idx,
            )
            tests.append(test_code)

        print(f"payload tests: \n{tests}")

        test_file = "\n\n\n".join(tests).rstrip() + "\n"
        return test_file

    def recover_message(self, stego_text: str) -> Any:
        """
        Undo the paraphrase attack by letting the LLM re-order tests by importance,
        then decode the payload from all tests after the first H entries.
        """
        if self.message_length is None or self.error_encoded_length is None:
            raise ValueError(
                "No encoded message metadata available; run hide_message first."
            )

        if self.message_length == 0:
            decoded_bits = self.ecc.decode([], self.error_encoded_length)
            return self.encoder.decode(decoded_bits)

        sorted_tests = self._sort_tests_by_priority(stego_text)
        payload_tests = sorted_tests[self.key : self.key + self.message_length]
        if len(payload_tests) < self.message_length:
            raise ValueError(
                "Insufficient payload tests recovered from the stego file."
            )

        embeddings = get_embeddings_in_batch(self.client, payload_tests)
        hashed_chunks = [self.hash_fn(emb.reshape(1, -1)) for emb in embeddings]
        decoded_bits = self.ecc.decode(hashed_chunks, self.error_encoded_length)
        return self.encoder.decode(decoded_bits)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _plan_behaviors(self, problem: str, total_behaviors: int) -> BehaviorPlan:
        payload_needed = max(total_behaviors - self.key, 0)
        high_behaviors: List[str] = []
        medium_behaviors: List[str] = []
        low_behaviors: List[str] = []

        remaining_high = self.key
        high_attempts = 0
        while remaining_high > 0:
            high_attempts += 1
            if high_attempts > self.PLAN_RETRY_LIMIT * 5:
                raise ValueError("Could not obtain enough high-priority behaviors.")
            batch = min(remaining_high, self.MAX_BEHAVIORS_PER_REQUEST)
            plan_chunk = self._request_behavior_batch(
                problem=problem,
                total=batch,
                expected_high=batch,
                expected_payload=0,
            )
            new_items = plan_chunk.high[:batch]
            if not new_items:
                continue
            take = min(batch, len(new_items))
            high_behaviors.extend(new_items[:take])
            remaining_high -= take

        remaining_payload = payload_needed
        payload_attempts = 0
        while remaining_payload > 0:
            payload_attempts += 1
            if payload_attempts > self.PLAN_RETRY_LIMIT * 10:
                raise ValueError("Could not obtain enough medium/low behaviors.")
            batch = min(remaining_payload, self.MAX_BEHAVIORS_PER_REQUEST)
            plan_chunk = self._request_behavior_batch(
                problem=problem,
                total=batch,
                expected_high=0,
                expected_payload=batch,
            )
            available_medium = plan_chunk.medium
            available_low = plan_chunk.low
            if not available_medium and not available_low:
                continue

            take_medium = min(remaining_payload, len(available_medium))
            medium_behaviors.extend(available_medium[:take_medium])
            remaining_payload -= take_medium

            if remaining_payload > 0 and available_low:
                take_low = min(remaining_payload, len(available_low))
                low_behaviors.extend(available_low[:take_low])
                remaining_payload -= take_low

        return BehaviorPlan(
            high=high_behaviors,
            medium=medium_behaviors,
            low=low_behaviors,
        )

    def _request_behavior_batch(
        self,
        problem: str,
        total: int,
        expected_high: int,
        expected_payload: int,
    ) -> BehaviorPlan:
        prompt = UNIT_BEHAVIOR_PLAN.format(
            total_behaviors=total,
            high_count=expected_high,
            remaining_count=expected_payload,
        )
        errors: list[str] = []
        for _ in range(self.PLAN_RETRY_LIMIT):
            response = generate_response(
                client=self.client,
                conversation_history=f"HumanEval problem:\n{problem}",
                system_prompt=prompt,
                max_length=3500,
                temperature=0.2,
                decomp_mode=True,
            )
            try:
                plan = BehaviorPlan.from_json(response)
                return plan
            except ValueError as err:
                errors.append(f"{err}: {response!r}")

        raise ValueError(
            "Failed to parse behavior plan after multiple attempts. "
            + " | ".join(errors[-2:])
        )

    def _generate_test(
        self,
        problem: str,
        behavior: str,
        priority: str,
        existing_tests: Sequence[str],
        target_bits: np.ndarray | None,
        chunk_index: int | None,
    ) -> str:
        prohibited: List[str] = []
        history = self._build_conversation_history(problem, existing_tests)
        attempt = 0
        while True:
            system_prompt = UNIT_TEST_GENERATION.format(
                behavior_description=behavior,
                priority=priority,
                prohibited_tests=self._format_prohibited_tests(prohibited),
            )
            response = generate_response(
                client=self.client,
                conversation_history=history,
                system_prompt=system_prompt,
                max_length=800,
                temperature=1,
            ).strip()
            if not response:
                raise ValueError("Model returned an empty test case.")

            if target_bits is None:
                return response

            sampled_bits = self._hash_text(response)
            attempt += 1
            chunk_label = (
                f"[payload {chunk_index + 1}/{self.message_length}] "
                if self.message_length and chunk_index is not None
                else ""
            )
            print(f"behavior: {behavior}")
            print(
                f"{chunk_label}attempt {attempt}: ",
                f"hash={sampled_bits.tolist()} target={target_bits.tolist()}",
            )
            print(f"response:\n{response}")
            if np.array_equal(sampled_bits, target_bits):
                return response

            if response not in prohibited:
                prohibited.append(response)

    def _build_conversation_history(
        self, problem: str, existing_tests: Sequence[str]
    ) -> List[str]:
        history = [f"HumanEval problem statement:\n{problem.strip()}"]

        if existing_tests:
            rendered_tests = "\n\n".join(existing_tests)
            history.append("Existing pytest tests for this problem:\n" + rendered_tests)
        return history

    def _format_prohibited_tests(self, prohibited: Sequence[str]) -> str:
        if not prohibited:
            return "None."
        return "\n\n".join(prohibited)

    def _hash_text(self, test_code: str) -> np.ndarray:
        embedding = (
            self.client.embeddings.create(
                input=[test_code],
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )
        return self.hash_fn(np.array(embedding).reshape(1, -1))

    def _sort_tests_by_priority(self, test_file: str) -> List[str]:
        response = generate_response(
            client=self.client,
            conversation_history=test_file,
            system_prompt=UNIT_TEST_SORT,
            max_length=2000,
            temperature=0,
            decomp_mode=True,
        )
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as err:
            raise ValueError("Failed to decode sorted tests JSON") from err

        tests = parsed.get("tests", [])
        if not isinstance(tests, list):
            raise ValueError("Sorted tests JSON missing 'tests' list.")
        return [
            test.strip() for test in tests if isinstance(test, str) and test.strip()
        ]
