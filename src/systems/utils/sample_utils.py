from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from ..config.constants import BacktrackConfig
from .get_embedding import get_embeddings_in_batch
from .new_text import generate_response


@dataclass
class SampleResult:
    success: bool
    matches: list[tuple[str, np.ndarray]] = field(default_factory=list)
    attempts_used: int = 0

    @property
    def primary(self) -> tuple[str, np.ndarray] | None:
        return self.matches[0] if self.matches else None


@dataclass
class StepChoice:
    matches: list[tuple[str, np.ndarray]]
    current_index: int = 0

    @property
    def message(self) -> str:
        return self.matches[self.current_index][0]

    @property
    def embedding(self) -> np.ndarray:
        return self.matches[self.current_index][1]

    def has_alternatives(self) -> bool:
        return self.current_index < len(self.matches) - 1

    def use_next_alternative(self) -> None:
        if not self.has_alternatives():
            raise IndexError("No more alternatives available")
        self.current_index += 1


class EncodingError(Exception):
    pass


def simple_prompt_builder(
    history: list[str], system_prompt: str, covered: set[str]
) -> tuple[str, str]:
    return system_prompt, " ".join(history)


def check_exact_duplicate(response: str, existing: set[str]) -> bool:
    return response in existing


def check_near_duplicate(response: str, existing: set[str]) -> bool:
    threshold = 0.8

    def jaccard_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
        if response in existing:
            return True

    new_words = set(response.lower().split())
    if not new_words:
        return False
    return any(
        jaccard_similarity(new_words, set(item.lower().split())) > threshold
        for item in existing
        if item
    )


def update_history_as_list(
    initial_history: list[str], steps: list["StepChoice"]
) -> list[str]:
    return initial_history + [choice.message for choice in steps]


class Sampler(Protocol):
    def sample(
        self,
        client: Any,
        desired_bits: np.ndarray,
        history: Any,
        hash_fn: Callable[[np.ndarray], np.ndarray],
        system_prompt: str,
        max_length: int,
        temperature: float,
        max_attempts: int,
        collect_alternatives: bool,
        max_alternatives: int,
    ) -> SampleResult: ...


class RejectionSampler:
    def __init__(
        self,
        prompt_builder: Callable[[Any, str, set[str]], tuple[str, str]] | None = None,
        duplicate_checker: Callable[[str, set[str]], bool] | None = None,
        response_cleaner: Callable[[str, Any], str] | None = None,
        count_duplicates: bool = False,
    ):
        self.build_prompt: Callable[[Any, str, set[str]], tuple[str, str]] = (
            prompt_builder or simple_prompt_builder
        )
        self.check_duplicate: Callable[[str, set[str]], bool] = (
            duplicate_checker or check_exact_duplicate
        )
        self.clean_response: Callable[[str, Any], str] | None = response_cleaner
        self.count_duplicates = count_duplicates

    def sample(
        self,
        client: Any,
        desired_bits: np.ndarray,
        history: Any,
        hash_fn: Callable[[np.ndarray], np.ndarray],
        system_prompt: str,
        max_length: int,
        temperature: float = 0.5,
        max_attempts: int = 50,
        collect_alternatives: bool = True,
        max_alternatives: int = 3,
    ) -> SampleResult:
        matches: list[tuple[str, np.ndarray]] = []
        attempts = 0
        covered: set[str] = set()

        while attempts < max_attempts:
            system_prompt_mod, prompt = self.build_prompt(
                history, system_prompt, covered
            )
            response = generate_response(
                prompt, system_prompt_mod, max_length, temperature
            )

            if self.clean_response:
                response = self.clean_response(response, history)

            if self.check_duplicate(response, covered):
                covered.add(response)
                if self.count_duplicates:
                    attempts += 1
                continue

            attempts += 1
            covered.add(response)

            embeddings = get_embeddings_in_batch(client, [response])
            embedding = np.array(embeddings[0]).reshape(1, -1)
            sampled_bits = hash_fn(embedding)

            if np.array_equal(sampled_bits, desired_bits):
                matches.append((response, embedding))
                if not collect_alternatives or len(matches) >= max_alternatives:
                    return SampleResult(
                        success=True, matches=matches, attempts_used=attempts
                    )

        return SampleResult(
            success=len(matches) > 0, matches=matches, attempts_used=attempts
        )


class BacktrackingEncoder:
    def __init__(
        self,
        sampler: Sampler | None = None,
        config: BacktrackConfig | None = None,
        history_updater: Callable[[Any, list[StepChoice]], Any] | None = None,
    ):
        self.sampler: Sampler = sampler or RejectionSampler()
        self.config: BacktrackConfig = config or BacktrackConfig()
        self.update_history: Callable[[Any, list[StepChoice]], Any] = (
            history_updater or update_history_as_list
        )

    def encode(
        self,
        client: Any,
        chunks: list[np.ndarray],
        initial_history: Any,
        hash_fn: Callable[[np.ndarray], np.ndarray],
        system_prompt: str,
        max_length: int,
        temperature: float = 0.5,
    ) -> tuple[list[str], list[np.ndarray]]:
        steps: list[StepChoice] = []
        i = 0
        backtracks_used = 0

        while i < len(chunks):
            history = self.update_history(initial_history, steps)
            collect_alternatives = (
                i != len(chunks) - 1
            ) and self.config.collect_alternatives
            max_attempts = (
                int(self.config.max_attempts_per_step * 1.5)
                if i == 0
                else self.config.max_attempts_per_step
            )

            result = self.sampler.sample(
                client=client,
                desired_bits=chunks[i],
                history=history,
                hash_fn=hash_fn,
                system_prompt=system_prompt,
                max_length=max_length,
                temperature=temperature,
                max_attempts=max_attempts,
                collect_alternatives=collect_alternatives,
                max_alternatives=self.config.max_alternatives,
            )

            if result.success:
                self._log_success(i, len(chunks), result)
                if i < len(steps):
                    steps[i] = StepChoice(result.matches)
                else:
                    steps.append(StepChoice(result.matches))
                i += 1
            else:
                backtracks_used += 1
                if backtracks_used > self.config.max_backtracks:
                    raise EncodingError(
                        f"Exceeded maximum backtracks ({self.config.max_backtracks})"
                    )

                target = self._find_backtrack_target(steps)
                if target < 0:
                    raise EncodingError(
                        f"No backtrack target available at position {i}"
                    )

                self._log_backtrack(i, target, backtracks_used)
                steps[target].use_next_alternative()
                steps = steps[: target + 1]
                i = target + 1

        return [choice.message for choice in steps], [
            choice.embedding for choice in steps
        ]

    def _find_backtrack_target(self, steps: list[StepChoice]) -> int:
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].has_alternatives():
                return i
        return -1

    def _log_success(self, position: int, total: int, result: SampleResult) -> None:
        alt_info = (
            f" (+{len(result.matches) - 1} alternatives)"
            if len(result.matches) > 1
            else ""
        )
        print(
            f"[{position + 1}/{total}] Encoded in {result.attempts_used} attempts{alt_info}"
        )

    def _log_backtrack(self, from_pos: int, to_pos: int, total_backtracks: int) -> None:
        print(f"[Backtrack #{total_backtracks}] {from_pos + 1} -> {to_pos + 1}")


def sample_concurrent():
    raise NotImplementedError
