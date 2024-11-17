import openai
from error_correction import RepetitionCode, ConvolutionalCode
from hash_functions import RandomProjectionHash, PCAHash, OracleHash
from steg import encode
from get_embedding import get_embeddings_in_batch
from paraphrase import paraphrase_message
from encoder import Encoder, StandardEncoder
from typing import Any
from simulation import Simulator

class StegSystem:
    def __init__(self, 
                 client,
                 hash_function,
                 error_correction,
                 encoder: Encoder = None,
                 system_prompt=None,
                 chunk_length=200,
                 simulator=None):
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or StandardEncoder()
        self.system_prompt = system_prompt
        self.chunk_length = chunk_length
        self.simulator = simulator
        
        # Get hash output length
        self.hash_output_length = getattr(hash_function, 'output_length')
        
        if self.simulator and not isinstance(hash_function, OracleHash):
            raise ValueError("Simulation mode can only be used with OracleHash, "
                           f"not {type(hash_function).__name__}")
        
    def hide_message(self, data: Any, history):
        # Get raw bits from encoder
        m_bits = self.encoder.encode(data)
        
        # Let the ECC handle any necessary padding
        m_encoded = self.ecc.encode(m_bits)
        
        # Convert to chunks of size hash_output_length
        m_chunks = [m_encoded[i:i + self.hash_output_length] 
                    for i in range(0, len(m_encoded), self.hash_output_length)]
        
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
            max_length=self.chunk_length
        )
        paraphrases = [paraphrase_message(self.client, text) for text in cover_text]
        return paraphrases
        
    def recover_message(self, stego_texts):
        if self.simulator:
            embeddings = [self.simulator.get_embedding(text) for text in stego_texts]
            bits_encoded = [self.hash_fn(emb, corrupt=True) for emb in embeddings]
        else:
            embeddings = get_embeddings_in_batch(self.client, stego_texts)
            bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        
        m_bits = self.ecc.decode(bits_encoded)
        
        return self.encoder.decode(m_bits)