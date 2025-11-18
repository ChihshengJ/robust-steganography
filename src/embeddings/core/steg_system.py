from typing import Any, List

from nltk.tokenize import sent_tokenize
import json

from ..config.system_prompts import STORY_SEGMENTATION, STORY_SEGMENTATION_NOCUE
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
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
        story_mode=False,
    ):
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.simulator = simulator
        self.error_encoded_length = None
        self.story_mode = story_mode

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

        # Let the ECC handle any necessary padding
        m_encoded: List[int] = self.ecc.encode(m_bits)
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
            
        # normally system only get the chunk length after hiding a message

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

    def recover_message(self, stego_text: str):
        # assuming stego_text is a block of text, not separated by sentence
        if self.simulator:
            embeddings = [self.simulator.get_embedding(text) for text in stego_text]
            bits_encoded = [self.hash_fn(emb, corrupt=True) for emb in embeddings]
        elif self.story_mode:
            chunks = self._rechunking_message(stego_text, self.error_encoded_length)
            print(
                f"original length: {self.error_encoded_length}, chunked_length: {len(chunks)}"
            )
            embeddings = get_embeddings_in_batch(self.client, chunks)
            bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        else:
            chunks = sent_tokenize(stego_text)
            print(
                f"original length: {self.error_encoded_length}, chunked_length: {len(chunks)}"
            )
            embeddings = get_embeddings_in_batch(self.client, stego_text)
            bits_encoded = [self.hash_fn(emb) for emb in embeddings]

        m_bits = self.ecc.decode(bits_encoded, self.error_encoded_length)

        return self.encoder.decode(m_bits)

    def set_chunk_length(self, chunk_length):
        self.error_encoded_length = chunk_length
        return

    def _rechunking_message(self, stego_text: str, chunk_length: int) -> list[str]:
        # system_prompt = STORY_SEGMENTATION.format(chunk_length=chunk_length)
        system_prompt = STORY_SEGMENTATION_NOCUE
        # print(f"system_prompt: {system_prompt}")
        response = generate_response(
            self.client,
            conversation_history=f"The story: {stego_text}",
            system_prompt=system_prompt,
            max_length=5000,
            temperature=0.0,
            # top_p=0.6,
            decomp_mode=True
        )
        # chunks = [chunk.strip() for chunk in chunks.split("[sep]")]
        print(f"========chunks=======\n{response}")
        chunks = json.loads(response)["events"]
        # print(f"chunk length: {len(chunks)}, actual elgnth: {chunk_length}")
        return chunks
