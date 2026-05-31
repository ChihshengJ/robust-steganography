from abc import ABC, abstractmethod
from typing import Any

from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction


class StegSystem(ABC):
    """
    Minimal abstract base class for steganography systems.

    A steganography system grows a payload into cover text from a ``seed`` and
    recovers it again. Sub-classes must implement:
    - hide_message: encode data into stego text, conditioned on ``seed``
    - recover_message: decode data back out of stego text

    The ``seed`` is the generative anchor of a stego message (a question, a
    story premise, or a corpus index) — not merely surrounding context but the
    thing a payload is grown from. The name is kept deliberately to match the
    paper.

    Embedding-hash systems (sentence/oracle/legacy) extend the richer
    :class:`EmbeddingStegSystem` base instead; the one-shot paper systems
    (Story, TopicQA, LitReview) inherit from this class directly.
    """

    def __init__(
        self,
        client: Any,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
    ) -> None:
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.hash_output_length: int = getattr(hash_function, "output_length")
        self._error_encoded_length: int | None = None

    @property
    def error_encoded_length(self) -> int | None:
        return self._error_encoded_length

    @error_encoded_length.setter
    def error_encoded_length(self, value: int | None) -> None:
        self._error_encoded_length = value

    def _encode_to_chunks(self, data: Any) -> tuple[list[list[int]], int]:
        """
        Encode data to bit chunks.

        Returns:
            Tuple of (chunks, encoded_length) where chunks are padded to hash_output_length
        """
        m_bits = self.encoder.encode(data)
        m_encoded = self.ecc.encode(m_bits)
        encoded_length = len(m_encoded)

        # Split into chunks of hash_output_length
        chunks = [
            m_encoded[i : i + self.hash_output_length]
            for i in range(0, len(m_encoded), self.hash_output_length)
        ]

        # Pad final chunk if needed
        chunks = [
            chunk + [0] * (self.hash_output_length - len(chunk)) for chunk in chunks
        ]

        return chunks, encoded_length

    @abstractmethod
    def hide_message(self, data: Any, seed: Any, **kwargs) -> str:
        """Grow data into cover text, conditioned on ``seed``."""
        ...

    @abstractmethod
    def recover_message(self, stego_text: str) -> Any:
        """Decode data from stego text."""
        ...
