from typing import Any, List

from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.steg import encode
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction, OracleHash


class StegSystem:
    def __init__(
        self,
        client,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        system_prompt: str | None = None,
        max_length: int = 200,
        simulator=None,
    ):
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.simulator = simulator
        self.error_encoded_length = None

        # Get hash output length
        self.hash_output_length = getattr(hash_function, "output_length")

        if self.simulator and not isinstance(hash_function, OracleHash):
            raise ValueError(
                "Simulation mode can only be used with OracleHash, "
                f"not {type(hash_function).__name__}"
            )

    def hide_message(self, data: Any, history) -> List[str]:
        # Get raw bits from encoder
        m_bits: list[int] = self.encoder.encode(data)
        # print(m_bits)

        # Let the ECC handle any necessary padding
        m_encoded: List[int] = self.ecc.encode(m_bits)
        # print(m_encoded)
        self.error_encoded_length = len(m_encoded)

        # Convert to chunks of size hash_output_length
        m_chunks = [
            m_encoded[i : i + self.hash_output_length]
            for i in range(0, len(m_encoded), self.hash_output_length)
        ]
        # padding
        m_chunks = [
            nested + [0] * (self.hash_output_length - len(nested))
            for nested in m_chunks
        ]
        # print(m_chunks)

        if self.simulator:
            cover_texts = []
            for desired_bits in m_chunks:
                while True:
                    text = self.simulator.generate_dummy_text()
                    embedding = self.simulator.get_embedding(text)
                    hash_bits = self.hash_fn(embedding)
                    if all(h == d for h, d in zip(hash_bits, desired_bits)):
                        cover_texts.append(text)
                        break
            return cover_texts

        # Normal mode - use real API calls
        cover_text = encode(
            self.client,
            m_chunks,
            history,
            self.hash_fn,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
        )
        return cover_text

    def recover_message(self, stego_texts):
        if self.simulator:
            embeddings = [self.simulator.get_embedding(text) for text in stego_texts]
            bits_encoded = [self.hash_fn(emb, corrupt=True) for emb in embeddings]
        else:
            embeddings = get_embeddings_in_batch(self.client, stego_texts)
            bits_encoded = [self.hash_fn(emb) for emb in embeddings]

        m_bits = self.ecc.decode(bits_encoded, self.error_encoded_length)

        return self.encoder.decode(m_bits)
