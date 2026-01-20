import json
from typing import Any

import numpy as np

from ..config.constants import BacktrackConfig
from ..config.system_prompts import STORY_SEGMENTATION_NOCUE
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from ..utils.sample_utils import BacktrackingEncoder, RejectionSampler
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction
from .steg_system import StegSystem


class StoryStegSystem(StegSystem):
    """
    Generate event based creative story plot for stego texts.
    """

    def __init__(
        self,
        client: Any,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        system_prompt: str,
        encoder: Encoder | None = None,
        max_length: int = 200,
        segmentation_prompt: str | None = None,
        backtrack_config: BacktrackConfig | None = None,
    ) -> None:
        super().__init__(client, hash_function, error_correction, encoder)
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.segmentation_prompt = segmentation_prompt or STORY_SEGMENTATION_NOCUE
        self.backtrack_config = backtrack_config or BacktrackConfig()
        self._backtracking_encoder = BacktrackingEncoder(
            sampler=RejectionSampler(),
            config=self.backtrack_config,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: list[str],
        system_prompt: str,
        max_length: int = 200,
        temperature: float = 0.5,
        **kwargs,
    ) -> tuple[list[str], list]:
        cover_texts, embeddings = self._backtracking_encoder.encode(
            client=self.client,
            chunks=[np.array(lst) for lst in chunks],
            initial_history=history if isinstance(history, list) else [history],
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
        )
        return cover_texts, embeddings

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        print(f"Chunks to encode: {len(chunks)}")

        initial_history = [seed] if isinstance(seed, str) else seed

        cover_texts, _ = self.encode(
            chunks,
            initial_history,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
        )
        return " ".join(cover_texts)

    def recover_message(self, stego_text: str) -> Any:
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        chunks = self._segment_story(stego_text)
        print(
            f"Segmented into {len(chunks)} events, "
            f"expected: {self._error_encoded_length // self.hash_output_length}"
        )

        embeddings = get_embeddings_in_batch(self.client, chunks)
        return self._decode_from_embeddings(embeddings, self._error_encoded_length)

    def _segment_story(self, stego_text: str) -> list[str]:
        response = generate_response(
            self.client,
            prompt=f"The story: {stego_text}",
            system_prompt=self.segmentation_prompt.format(
                chunk_length=self._error_encoded_length
            ),
            max_length=5000,
            temperature=0,
            json_mode=True,
        )
        print(f"Segmentation response:\n{response}")

        parsed = json.loads(response)
        events = parsed.get("events", [])

        if not events:
            raise ValueError("Story segmentation returned no events")

        return [event.strip() for event in events if event.strip()]
