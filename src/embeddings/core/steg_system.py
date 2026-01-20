from abc import ABC, abstractmethod
from typing import Any, Sequence

from nltk.tokenize import sent_tokenize

from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.sample_utils import sample_concurrent
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction, OracleHash
from .simulation import Simulator


class StegSystem(ABC):
    """
    Abstract base class for steganography systems.

    Sub-classes must implement:
    - hide_message: Encode data into stego text
    - recover_message: Decode data from stego text
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

    def _decode_from_embeddings(
        self,
        embeddings: Sequence[Any],
        encoded_length: int,
    ) -> Any:
        """Decode data from a sequence of embeddings."""
        bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        m_bits = self.ecc.decode(bits_encoded, encoded_length)
        return self.encoder.decode(m_bits)

    @abstractmethod
    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str,
        max_length=200,
        **kwargs,
    ) -> Any:
        """Encode the chunks into template items."""
        ...

    @abstractmethod
    def hide_message(self, data, seed, **kwargs) -> str:
        """Encode data into cover text."""
        ...

    @abstractmethod
    def recover_message(self, stego_text: str) -> Any:
        """Decode data from stego text."""
        ...


class OracleStegSystem(StegSystem):
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

    def hide_message(self, data: Any, seed: str = "", **kwargs) -> str:
        """Generate simulated cover text encoding the data."""
        chunks, self._error_encoded_length = self._encode_to_chunks(data)

        cover_texts: list[str] = []
        for desired_bits in chunks:
            cover_texts.append(self._find_matching_text(desired_bits))

        return " ".join(cover_texts)

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


class SentenceStegSystem(StegSystem):
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
    ) -> None:
        super().__init__(client, hash_function, error_correction, encoder)
        self.system_prompt = system_prompt
        self.max_length = max_length

    def encode(
        self,
        chunks: list[list[int]],
        history: list[str],
        system_prompt: str = "You are having a casual conversation.",
        max_length=200,
        k: int = 5,
        **kwargs,
    ) -> list[str]:
        cover_text = []
        for chunk in chunks:
            response = sample_concurrent(
                client=self.client,
                desired_bits=chunk,
                history=history,
                hash_fn=self.hash_fn,
                temperature=0.5,
                system_prompt=system_prompt,
                k=k,
                max_length=max_length,
            )
            assert isinstance(response, list), "No response"
            history.append(response)
            cover_text.append(response)
        return cover_text

    def hide_message(self, data: Any, seed: list[str], **kwargs) -> str:
        """Generate cover text with sentences encoding the data."""
        chunks, self._error_encoded_length = self._encode_to_chunks(data)

        cover_texts = self.encode(
            chunks=chunks,
            history=seed,
            k=5,
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
