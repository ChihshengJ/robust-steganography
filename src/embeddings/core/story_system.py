from src.embeddings.config.system_prompts import STORY_SEGMENTATION
from nltk import sent_tokenize
import json
from typing import Any

import numpy as np

from ..config.constants import BacktrackConfig
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.sample_utils import BacktrackingEncoder, RejectionSampler
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction
from .steg_system import StegSystem


class StoryStegSystem(StegSystem):
    def __init__(
        self,
        client: Any,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        system_prompt: str,
        encoder: Encoder | None = None,
        max_length: int = 200,
        backtrack_config: BacktrackConfig | None = None,
        segmentation_model: str = "gpt-4.1",
    ) -> None:
        super().__init__(client, hash_function, error_correction, encoder)
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.backtrack_config = backtrack_config or BacktrackConfig()
        self.segmentation_model = segmentation_model
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
        temperature: float = 0.7,
        **kwargs,
    ) -> tuple[list[str], list]:
        return self._backtracking_encoder.encode(
            client=self.client,
            chunks=[np.array(lst) for lst in chunks],
            initial_history=history if isinstance(history, list) else [history],
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
        )

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

    def _segment_with_llm(self, text: str, n_chunks: int) -> list[str]:
        prompt = STORY_SEGMENTATION.format(n_chunks=n_chunks)

        response = self.client.chat.completions.create(
            model=self.segmentation_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        content = content.strip()
        print(f"    segmented: {content}")

        parsed = json.loads(content)

        if isinstance(parsed, dict) and "chunks" in parsed:
            chunks = parsed["chunks"]
        elif isinstance(parsed, list):
            chunks = parsed
        else:
            chunks = list(parsed.values())[0]

        if len(chunks) != n_chunks:
            print(f"WARNING: Sentence count mismatch ({len(chunks)} vs {n_chunks})")

        return [c.strip() for c in chunks]

    def recover_message(self, stego_text: str) -> Any:
        """Recover hidden message from stego text."""
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        expected_chunks = self._error_encoded_length // self.hash_output_length

        sentences = self._segment_with_llm(stego_text, expected_chunks)
        print(f"LLM segmentation: {expected_chunks} chunks recovered")
        embeddings = get_embeddings_in_batch(self.client, sentences)
        return self._decode_from_embeddings(embeddings, self._error_encoded_length)


    def recover_message_legacy(self, stego_text: str) -> Any:
        """Recover hidden message from stego text."""
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        expected_chunks = self._error_encoded_length // self.hash_output_length

        sentences = [s.strip() for s in sent_tokenize(stego_text) if s.strip()]
        print(f"Extracted {len(sentences)} sentences, expected {expected_chunks}")
        if len(sentences) != expected_chunks:
            print(
                f"WARNING: Sentence count mismatch ({len(sentences)} vs {expected_chunks})"
            )
        embeddings = get_embeddings_in_batch(self.client, sentences)
        return self._decode_from_embeddings(embeddings, self._error_encoded_length)
