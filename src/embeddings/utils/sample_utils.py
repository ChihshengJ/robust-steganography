from dataclasses import dataclass, field
from typing import Protocol, Any, Callable
import numpy as np
import concurrent.futures

from .get_embedding import get_embeddings_in_batch
from .new_text import generate_response
from ..config.constants import BacktrackConfig


@dataclass
class SampleResult:
    success: bool
    matches: list[tuple[str, np.ndarray]] = field(default_factory=list)
    attempts_used: int = 0

    @property
    def primary(self) -> tuple[str, np.ndarray] | None:
        return self.matches[0] if self.matches else None


class Sampler(Protocol):
    def sample(
        self,
        client: Any,
        desired_bits: np.ndarray,
        history: list[str],
        hash_fn: Callable[[np.ndarray], np.ndarray],
        system_prompt: str,
        max_length: int,
        temperature: float,
        max_attempts: int,
        collect_alternatives: bool,
        max_alternatives: int,
    ) -> SampleResult:
        ...


class RejectionSampler:
    def sample(
        self,
        client: Any,
        desired_bits: np.ndarray,
        history: list[str],
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
        prohibitions = []

        while attempts < max_attempts:
            if len(prohibitions):
                system_prompt = system_prompt + f'You must not include these outputs:\n {"\n".join(prohibitions)}'
            response = generate_response(
                client,
                history,
                system_prompt,
                max_length,
                temperature,
            )
            attempts += 1

            embeddings = get_embeddings_in_batch(client, [response])
            embedding = np.array(embeddings[0]).reshape(1, -1)
            sampled_bits = hash_fn(embedding)

            if np.array_equal(sampled_bits, desired_bits):
                matches.append((response, embedding))

                if not collect_alternatives or len(matches) >= max_alternatives:
                    return SampleResult(
                        success=True,
                        matches=matches,
                        attempts_used=attempts,
                    )
            prohibitions.append(response)

        return SampleResult(
            success=len(matches) > 0,
            matches=matches,
            attempts_used=attempts,
        )


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


class BacktrackingEncoder:
    def __init__(
        self,
        sampler: Sampler | None = None,
        config: BacktrackConfig | None = None,
    ):
        self.sampler = sampler or RejectionSampler()
        self.config = config or BacktrackConfig()

    def encode(
        self,
        client: Any,
        chunks: list[np.ndarray],
        initial_history: list[str],
        hash_fn: Callable[[np.ndarray], np.ndarray],
        system_prompt: str,
        max_length: int,
        temperature: float = 0.5,
    ) -> tuple[list[str], list[np.ndarray]]:
        choices: list[StepChoice] = []
        i = 0
        backtracks_used = 0

        while i < len(chunks):
            history = self._build_history(initial_history, choices)

            result = self.sampler.sample(
                client=client,
                desired_bits=chunks[i],
                history=history,
                hash_fn=hash_fn,
                system_prompt=system_prompt,
                max_length=max_length,
                temperature=temperature,
                max_attempts=self.config.max_attempts_per_step,
                collect_alternatives=self.config.collect_alternatives,
                max_alternatives=self.config.max_alternatives,
            )

            if result.success:
                self._log_success(i, len(chunks), result)
                if i < len(choices):
                    choices[i] = StepChoice(result.matches)
                else:
                    choices.append(StepChoice(result.matches))
                i += 1
            else:
                backtracks_used += 1
                if backtracks_used > self.config.max_backtracks:
                    raise EncodingError(
                        f"Exceeded maximum backtracks ({self.config.max_backtracks})"
                    )

                target = self._find_backtrack_target(choices, i)
                if target < 0:
                    raise EncodingError(
                        f"No backtrack target available at position {i}"
                    )

                self._log_backtrack(i, target, backtracks_used)
                choices[target].use_next_alternative()
                choices = choices[: target + 1]
                i = target + 1

        cover_texts = [choice.message for choice in choices]
        embeddings = [choice.embedding for choice in choices]
        return cover_texts, embeddings

    def _build_history(
        self,
        initial_history: list[str],
        choices: list[StepChoice],
    ) -> list[str]:
        return initial_history + [choice.message for choice in choices]

    def _find_backtrack_target(
        self,
        choices: list[StepChoice],
        current_position: int,
    ) -> int:
        for i in range(len(choices) - 1, -1, -1):
            if choices[i].has_alternatives():
                return i
        return -1

    def _log_success(self, position: int, total: int, result: SampleResult) -> None:
        alt_info = ""
        if len(result.matches) > 1:
            alt_info = f" (+{len(result.matches) - 1} alternatives)"
        print(f"[{position + 1}/{total}] Encoded in {result.attempts_used} attempts{alt_info}")

    def _log_backtrack(self, from_pos: int, to_pos: int, total_backtracks: int) -> None:
        print(f"[Backtrack #{total_backtracks}] {from_pos} -> {to_pos}")



### Legacy ###
def sample_concurrent(
    client,
    desired_bits,
    history,
    hash_fn,
    temperature,
    system_prompt,
    k=4,
    max_length=200,
) -> str | None:
    sampled_bits = np.nan

    while not np.array_equal(sampled_bits, desired_bits):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Step 1: Parallelize `generate_response`
            response_futures = [
                executor.submit(
                    generate_response,
                    client,
                    history,
                    system_prompt,
                    max_length,
                    temperature,
                )
                for _ in range(k)
            ]
            responses = [
                future.result()
                for future in concurrent.futures.as_completed(response_futures)
            ]

            # Step 2: Get embeddings in batch
            embeddings = get_embeddings_in_batch(client, responses)

            # Process embeddings
            for message, emb in zip(responses, embeddings):
                emb = np.array(emb).reshape(1, -1)
                sampled_bits = hash_fn(emb)

                print("message:", message)
                print(f"sampled_bits: {sampled_bits}, desired bits: {desired_bits}")

                #! Ensure matching shapes for all combinations of inputs and settings
                if np.array_equal(sampled_bits, desired_bits):
                    return message

