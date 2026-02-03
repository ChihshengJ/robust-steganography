import concurrent.futures
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
        use_prohibitions: bool,
    ) -> SampleResult: ...


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
        use_prohibitions: bool = False,
    ) -> SampleResult:
        matches: list[tuple[str, np.ndarray]] = []
        attempts = 0
        prohibitions: set[str] = set()

        while attempts < max_attempts:
            curr_system_prompt = system_prompt

            if prohibitions:
                if use_prohibitions:
                    prohibition_list = "\n".join(
                        f"- {p[:150]}..." if len(p) > 150 else f"- {p}"
                        for p in list(prohibitions)[-10:]
                    )
                    curr_system_prompt += f"\n\nDo NOT generate any of these previously attempted outputs:\n{prohibition_list}"

            prompt = " ".join(history)
            # print(f"prompt:\n   {prompt}...")

            response = generate_response(
                client,
                prompt,
                curr_system_prompt,
                max_length,
                temperature,
            )
            attempts += 1

            if response in prohibitions:
                # print(f"  [Attempt {attempts}] Duplicate response, skipping...")
                continue

            embeddings = get_embeddings_in_batch(client, [response])
            embedding = np.array(embeddings[0]).reshape(1, -1)
            sampled_bits = hash_fn(embedding)

            # print(
            #     f"  [Attempt {attempts}] desired_bits: {desired_bits}, sampled_bits: {sampled_bits}"
            # )

            if np.array_equal(sampled_bits, desired_bits):
                matches.append((response, embedding))
                prohibitions.add(response)  # Prevent finding same match again

                if not collect_alternatives or len(matches) >= max_alternatives:
                    return SampleResult(
                        success=True,
                        matches=matches,
                        attempts_used=attempts,
                    )
            else:
                prohibitions.add(response)

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
        use_prohibitions: bool = False,
    ) -> tuple[list[str], list[np.ndarray]]:
        steps: list[StepChoice] = []
        i = 0
        backtracks_used = 0

        while i < len(chunks):
            history = self._build_history(initial_history, steps)

            collect_alternatives = (
                False if i == (len(chunks) - 1) else self.config.collect_alternatives
            )
            max_attempts = self.config.max_attempts_per_step
            if i == 0:
                max_attempts = int(max_attempts * 1.5)

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
                use_prohibitions=use_prohibitions,
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

                target = self._find_backtrack_target(steps, i)
                if target < 0:
                    raise EncodingError(
                        f"No backtrack target available at position {i}"
                    )

                self._log_backtrack(i, target, backtracks_used)
                steps[target].use_next_alternative()
                steps = steps[: target + 1]
                i = target + 1

        cover_texts = [choice.message for choice in steps]
        embeddings = [choice.embedding for choice in steps]
        return cover_texts, embeddings

    def _build_history(
        self,
        initial_history: list[str],
        steps: list[StepChoice],
    ) -> list[str]:
        return initial_history + [choice.message for choice in steps]

    def _find_backtrack_target(
        self,
        steps: list[StepChoice],
        current_position: int,
    ) -> int:
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].has_alternatives():
                return i
        return -1

    def _log_success(self, position: int, total: int, result: SampleResult) -> None:
        alt_info = ""
        if len(result.matches) > 1:
            alt_info = f" (+{len(result.matches) - 1} alternatives)"
        print(
            f"[{position + 1}/{total}] Encoded in {result.attempts_used} attempts{alt_info}"
        )

    def _log_backtrack(self, from_pos: int, to_pos: int, total_backtracks: int) -> None:
        print(f"[Backtrack #{total_backtracks}] {from_pos + 1} -> {to_pos + 1}")


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
