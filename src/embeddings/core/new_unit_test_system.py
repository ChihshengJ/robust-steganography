import json
from dataclasses import dataclass, fields
from typing import Any, Iterable, List, Sequence, Tuple

from embeddings import StegSystem
import numpy as np

from ..config.system_prompts import (
    UNIT_TEST_BEHAVIOR_CONTINUATION,
    UNIT_TEST_GENERATION_ALT,
    UNIT_TEST_H_BEHAVIOR_GENERATION,
    UNIT_TEST_SORT,
    UNIT_TEST_TO_BEHAVIOR,
)
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction, GenerationContext, MajorityVoteHash


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

    def items(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)

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

    def update_behaviors(self, text: str):
        data = self._parse_payload(text)
        key = next(iter(data.keys()))
        new_items = [
            item.strip() for item in data[key] if isinstance(item, str) and item.strip()
        ]
        getattr(self, key).extend(new_items)


class UnitTestSystem(StegSystem):
    """
    encode/decode covert bits inside HumanEval-style test suites.

    H (self.key) designates the number of high-priority tests that are emitted before any
    steganographic encoding occurs. The remaining tests each encode one chunk of bits.
    """

    MAX_BEHAVIORS_PER_REQUEST = 50
    PLAN_RETRY_LIMIT = 30

    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        bit_prefix: Sequence[int] | None = None,
        majority_vote: bool = False,
    ):
        if key < 0:
            raise ValueError(
                "key must be non-negative (represents high priority behaviors)"
            )
        super().__init__(client, hash_function, error_correction, encoder)
        self.error_encoded_length: int | None = None
        self.hash_output_length = (
            1 if majority_vote else getattr(hash_function, "output_length")
        )
        self.key = key
        self.message_length: int = 0
        self.majority_vote = majority_vote

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        """
        Given a HumanEval problem statement, generate:
          1. H high-priority tests that anchor the suite behaviorally.
          2. Additional tests whose embeddings hash to the encoded message bits.

        Args:
            data: the payload
            seed: the HumanEval problem
        """

        # Preprocessing secret bits
        m_bits = self.encoder.encode(data)
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

        # Generate a list of key behaviors for generation
        print(f"article: \n{seed}")
        print(f"key: {self.key}; message_length: {self.message_length}")
        behavior_plan = self._plan_high_behaviors(problem=seed, key=self.key)

        # Generate behaviors for steganographic payload.
        behavior_plan = self._generate_behaviors(seed, chunks, behavior_plan)

        # Generate tests for behaviors
        tests = self._generate_tests(seed, behavior_plan)

        stego = f"Problem:\n{seed}\nTests:\n{'[sep]'.join(tests).rstrip()}\n"
        return stego

    def recover_message(self, stego_text: str) -> Any:
        """
        Undo the paraphrase attack by letting the LLM re-order tests by importance,
        then decode the payload behaviors from all tests after first {key} tests.
        """
        if self.message_length is None or self.error_encoded_length is None:
            raise ValueError(
                "No encoded message metadata available; run hide_message first."
            )
        if self.message_length == 0:
            decoded_bits = self.ecc.decode([], self.error_encoded_length)
            return self.encoder.decode(decoded_bits)

        problem, tests, *_ = stego_text.split("Tests:")

        sorted_tests = self._sort_tests_by_priority(tests)
        payload_tests = sorted_tests[
            self.key : self.key + self.message_length
        ]

        if len(payload_tests) < self.message_length:
            raise ValueError(
                "Insufficient payload tests recovered from the stego file."
            )
        print(f"sorted tests: {payload_tests}")

        # Get behaviors from tests
        behaviors = self._translate_tests_to_behaviors(problem, sorted_tests)
        payload_behaviors = behaviors[self.key: self.key + self.message_length]
        print(f"translated behaviors: {payload_behaviors}")

        # Get embeddings from behavior descriptions
        embeddings = get_embeddings_in_batch(self.client, payload_behaviors)
        if not self.majority_vote:
            hashed_chunks = [self.hash_fn(emb.reshape(1, -1)) for emb in embeddings]
        else:
            assert isinstance(self.hash_fn, MajorityVoteHash), "Must use MajorityVoteHash to use this mode"
            hashed_chunks  = []
            payload_contexts = self._parse_stego_text_to_contexts(problem, behaviors)[self.key: self.key + self.message_length]
            for emb, ctx in zip(embeddings, payload_contexts):
                self.hash_fn.calibrate(ctx)
                hashed_chunks.append(self.hash_fn(emb.reshape(1, -1)))
                
        decoded_bits = self.ecc.decode(hashed_chunks, self.error_encoded_length)
        return self.encoder.decode(decoded_bits)

    def _plan_high_behaviors(self, problem: str, key: int) -> BehaviorPlan:
        """
        Plan {key} high priority behaviors.
        """
        system_prompt = UNIT_TEST_H_BEHAVIOR_GENERATION.format(key=key)
        plan = BehaviorPlan([], [], [])
        for _ in range(self.PLAN_RETRY_LIMIT):
            response = generate_response(
                client=self.client,
                prompt=f"HumanEval problem:\n{problem}",
                system_prompt=system_prompt,
                max_length=500,
                temperature=0,
                json_mode=True,
            )
            print(f"plan high behavior response:\n {response}")
            plan.update_behaviors(response)
            if len(plan.high) == key:
                return plan
        raise ValueError("Failed to parse behavior plan after multiple attempts. ")

    def _generate_behaviors(
        self,
        problem: str,
        chunks: list[list[int]],
        existing_plan: BehaviorPlan,
    ) -> BehaviorPlan:
        for idx, chunk in enumerate(chunks):
            if chunk is None:
                raise ValueError("No chunk specified")
            prohibited: list[str] = []
            history = self._build_conversation_history(problem, existing_plan)
            attempt = 0
            while True:
                system_prompt = UNIT_TEST_BEHAVIOR_CONTINUATION.format(
                    prohibited_behaviors=prohibited
                )
                response = generate_response(
                    client=self.client,
                    prompt=history,
                    system_prompt=system_prompt,
                    max_length=500,
                    temperature=1,
                ).strip()
                if not response:
                    raise ValueError("Model returned an empty behavior description.")
                if not self.majority_vote:
                    sampled_bits = self._hash_text(response)
                else:
                    sampled_bits = self._hash_text(response, history, system_prompt)
                print(f"raw response: {response}")
                attempt += 1

                chunk_label = (
                    f"[payload {idx + 1}/{self.message_length}] "
                    if self.message_length and idx is not None
                    else ""
                )
                print(
                    f"{chunk_label}attempt {attempt}: ",
                    f"hash={sampled_bits.tolist()} target={chunk}",
                )

                if np.array_equal(sampled_bits, chunk):
                    existing_plan.update_behaviors(response)
                    history = self._build_conversation_history(problem, existing_plan)
                    break
                if response not in prohibited:
                    prohibited.append(response.split("\n")[0])
        return existing_plan

    def _generate_tests(self, problem: str, existing_plan: BehaviorPlan) -> list[str]:
        if self.message_length == 0:
            raise ValueError("message length == 0, system not initialized correctly")
        # Build a single prompt from a behavior plan
        behaviors = [b for _, behaviors in existing_plan.items() for b in behaviors]
        prompt = f"HumanEval problem:\n{problem}\nBehavior plan (list of behaviors for testing):\n{'\n'.join(behaviors)}"
        system_prompt = UNIT_TEST_GENERATION_ALT.format(
            length=self.message_length + self.key
        )
        response = generate_response(
            client=self.client,
            prompt=prompt,
            system_prompt=system_prompt,
            max_length=3000,
            temperature=0,
        )
        tests = response.strip().split("[sep]")
        if len(tests) != (self.message_length + self.key):
            raise ValueError("Tests does not line up with the behaviors")
        return tests

    def _build_conversation_history(self, problem: str, curr_plan: BehaviorPlan) -> str:
        history = f"HumanEval problem statement:\n{problem.strip()}\n"
        existing_behaviors = [
            b for _, behaviors in curr_plan.items() for b in behaviors
        ]

        if existing_behaviors:
            rendered_tests = "\n\n".join(existing_behaviors)
            history += "Existing pytest tests for this problem:\n" + rendered_tests
        return history

    def _format_prohibited_tests(self, prohibited: Sequence[str]) -> str:
        if not prohibited:
            return "None."

        return "\n".join(prohibited)

    def _hash_text(self, text: str, *args) -> np.ndarray:
        embedding = (
            self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )
        if not self.majority_vote:
            return self.hash_fn(embedding)

        # Use the majority vote hash
        assert isinstance(self.hash_fn, MajorityVoteHash), "only SampledPCAHash can use majority vote mode"
        ctx = GenerationContext(
            self.client,
            args[1],
            args[2]
        )
        self.hash_fn.calibrate(ctx)
        return self.hash_fn(embedding)

    def _sort_tests_by_priority(self, test_file: str) -> List[str]:
        response = generate_response(
            client=self.client,
            prompt=test_file,
            system_prompt=UNIT_TEST_SORT,
            max_length=2000,
            temperature=0,
            json_mode=True,
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

    def _translate_tests_to_behaviors(
        self, problem: str, tests: list[str]
    ) -> list[str]:
        system_prompt = UNIT_TEST_TO_BEHAVIOR
        prompt = f"HumanEval problem:\n {problem}\nList of unit tests:\n{tests}"
        response = generate_response(
            client=self.client,
            prompt=prompt,
            system_prompt=system_prompt,
            max_length=1000,
            temperature=0,
            json_mode=True,
        )
        return json.loads(response.strip())["behaviors"]

    def _parse_stego_text_to_contexts(self, problem, behaviors) -> list[GenerationContext]:
        contexts = []
        system_prompt = UNIT_TEST_BEHAVIOR_CONTINUATION
        for be in behaviors:
            history = f"HumanEval problem statement:\n{problem.strip()}\n Behaviors to test:\n"
            history += be
            contexts.append(GenerationContext(
                client=self.client,
                history=history,
                system_prompt=system_prompt,
            ))
        return contexts

