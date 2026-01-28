from typing import Any

import numpy as np
from nltk.tokenize import sent_tokenize

from ..config.constants import BacktrackConfig
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.sample_utils import BacktrackingEncoder, RejectionSampler
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction
from .steg_system import StegSystem


class StoryStegSystem(StegSystem):
    """
    Generate event-based creative story plot for stego texts.

    Encoding: Each story event (mostly sentence) encodes one chunk of bits via its embedding.
    Recovery: Simple sentence tokenization extracts events, then embeddings are hashed.
    """

    def __init__(
        self,
        client: Any,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        system_prompt: str,
        encoder: Encoder | None = None,
        max_length: int = 200,
        backtrack_config: BacktrackConfig | None = None,
    ) -> None:
        super().__init__(client, hash_function, error_correction, encoder)
        self.system_prompt = system_prompt
        self.max_length = max_length
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
        temperature: float = 0.7,
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
            use_prohibitions=False,
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
        """
        Recover message using just sentence tokenization.
        """
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        expected_chunks = self._error_encoded_length // self.hash_output_length
        sentences = sent_tokenize(stego_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        print(f"Extracted {len(sentences)} sentences, expected {expected_chunks}")
        if len(sentences) != expected_chunks:
            print(
                f"WARNING: Sentence count mismatch ({len(sentences)} vs {expected_chunks})"
            )

        embeddings = get_embeddings_in_batch(self.client, sentences)
        return self._decode_from_embeddings(embeddings, self._error_encoded_length)
