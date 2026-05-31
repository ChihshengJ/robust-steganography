from abc import abstractmethod
from typing import Any, Sequence

import numpy as np
from nltk.tokenize import sent_tokenize

from ...config.constants import BacktrackConfig
from ...utils.get_embedding import get_embeddings_in_batch
from ...utils.sample_utils import (
    BacktrackingEncoder,
    RejectionSampler,
)
from ..encoder import Encoder
from ..error_correction import ErrorCorrection
from ..hash_functions import HashFunction, OracleHash
from ..steg_system import StegSystem
from .simulation import Simulator


class EmbeddingStegSystem(StegSystem):
    """
    Base class for embedding-hash steganography systems.

    Unlike the one-shot paper systems, these systems realize each payload
    chunk as a span of text (a sentence/step) whose *embedding hash* matches
    the target bits, found by rejection/backtracking sampling. Decoding
    re-embeds the spans and reads the hash bits back out.

    Sub-classes implement ``encode`` (the per-chunk sampling loop) on top of
    the minimal ``hide_message``/``recover_message`` contract from
    :class:`StegSystem`.
    """

    @abstractmethod
    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str,
        max_length: int = 200,
        **kwargs,
    ) -> Any:
        """Encode the chunks into spans whose embedding hashes match the bits."""
        ...

    def _decode_from_embeddings(
        self,
        embeddings: Sequence[Any],
        encoded_length: int,
    ) -> Any:
        """Decode data from a sequence of embeddings."""
        bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        m_bits = self.ecc.decode(bits_encoded, encoded_length)
        return self.encoder.decode(m_bits)


class OracleStegSystem(EmbeddingStegSystem):
    """
    Steganography system using a simulator for testing.

    Uses an oracle hash and simulated text generation for controlled experiments.
    """

    def __init__(
        self,
        client: Any,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        simulator: Simulator,
        encoder: Encoder | None = None,
    ) -> None:
        super().__init__(client, hash_function, error_correction, encoder)
        self.simulator = simulator

        # Validate hash function type
        if not isinstance(hash_function, OracleHash):
            raise TypeError(
                f"OracleStegSystem requires OracleHash, got {type(hash_function).__name__}"
            )

    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str = "",
        max_length: int = 200,
        **kwargs,
    ) -> list[str]:
        """Rejection-sample one matching dummy text per chunk."""
        return [self._find_matching_text(desired_bits) for desired_bits in chunks]

    def hide_message(self, data: Any, seed: str = "", **kwargs) -> str:
        """Generate simulated cover text encoding the data."""
        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        return " ".join(self.encode(chunks, seed))

    def _find_matching_text(self, desired_bits: list[int]) -> str:
        """Rejection sample until we find text with matching hash."""
        while True:
            text = self.simulator.generate_dummy_text()
            embedding = self.simulator.get_embedding(text)
            hash_bits = self.hash_fn(embedding)

            if list(hash_bits) == desired_bits:
                return text

    def recover_message(self, stego_text: str | list[str]) -> Any:
        """Recover message from simulated stego text."""
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        # Handle both string and pre-split list
        if isinstance(stego_text, str):
            chunks = stego_text.split()  # Assuming space-separated
        else:
            chunks = stego_text

        embeddings = [self.simulator.get_embedding(text) for text in chunks]

        # Oracle mode may use corrupt=True for testing error correction
        bits_encoded = [self.hash_fn(emb, corrupt=True) for emb in embeddings]
        m_bits = self.ecc.decode(bits_encoded, self._error_encoded_length)

        return self.encoder.decode(m_bits)


class SentenceStegSystem(EmbeddingStegSystem):
    """
    Steganography system using sentence-level encoding.

    Each sentence in the cover text encodes one chunk of bits.
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
            sampler=RejectionSampler(), config=self.backtrack_config
        )

        self._seed: str | None = None
        self._last_metadata: dict | None = None

    @property
    def seed(self) -> str | None:
        return self._seed

    @seed.setter
    def seed(self, value: str | None) -> None:
        self._seed = value

    def encode(
        self,
        chunks: list[list[int]],
        history: list[str],
        system_prompt: str = "You are having a casual conversation.",
        max_length=200,
        k: int = 2,
        temperature: float = 1,
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

    def hide_message(self, data: Any, seed: list[str], **kwargs) -> str:
        """
        Generate cover text with sentences encoding the data.

        Keeping the list of history the same as the original code.
        """

        chunks, self._error_encoded_length = self._encode_to_chunks(data)

        cover_texts, _ = self.encode(
            chunks=chunks,
            history=seed,
            k=2,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
        )

        return " ".join(cover_texts)

    def recover_message(self, stego_text: str) -> Any:
        """Recover message by tokenizing into sentences."""
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )

        chunks = sent_tokenize(stego_text)
        print(
            f"Sentence count: {len(chunks)}, expected chunks: {self._error_encoded_length // self.hash_output_length}"
        )

        embeddings = get_embeddings_in_batch(self.client, chunks)
        return self._decode_from_embeddings(embeddings, self._error_encoded_length)
