from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from openai import OpenAI

from embeddings.utils.new_text import generate_response


class HashFunction:
    def __init__(self):
        pass

    def _to_numpy_array(self, emb):
        return np.array(emb)

    # Ensure that the output is a 1D array of bits
    def __call__(self, emb):
        raise NotImplementedError("Hash function must implement __call__")

    def get_output_length(self):
        raise NotImplementedError("Hash function must implement get_output_length")


class RandomProjectionHash(HashFunction):
    def __init__(self, embedding_dim=3072, num_bits=1, seed=128):
        super().__init__()
        np.random.seed(seed)
        self.rand_matrix = np.random.randn(embedding_dim, num_bits)
        self.output_length = num_bits

    def __call__(self, emb):
        emb = self._to_numpy_array(emb)
        projection = emb @ self.rand_matrix
        hashes = (projection > 0).astype(int)
        return hashes.ravel()

    def get_output_length(self):
        return self.output_length


class NaivePCAHash(HashFunction):
    def __init__(self, pca_model, start=0, end=1):
        super().__init__()
        self.pca = pca_model
        self.start = start
        self.end = end
        self.output_length = end - start

    def __call__(self, emb):
        emb = self._to_numpy_array(emb)
        transformed = self.pca.transform(emb.reshape(1, -1))
        return (transformed[:, self.start : self.end] > 0).astype(int).ravel()

    def get_output_length(self):
        return self.output_length


class PCAHash(HashFunction):
    def __init__(self, pca_dir: str, start: int, end: int):
        super().__init__()
        self.components = np.load(
            f"{pca_dir}/pca_components.npy"
        )  # (n_components, embed_dim)
        self.mean = np.load(f"{pca_dir}/pca_mean.npy")  # (embed_dim, )
        self.thresholds = np.load(f"{pca_dir}/pca_thresholds.npy")  # (n_components, )
        self.n = self.components.shape[0]
        self.output_length = abs(start - end)
        assert start < end
        assert end <= self.n
        self.interval = (start, end)

    def __call__(self, emb):
        emb = self._to_numpy_array(emb)
        z = (emb - self.mean) @ self.components.T  # (n_components,)
        bits = (z > self.thresholds).astype(np.int8).ravel()
        # slice out the bits to use
        print(f"actual bits: {bits}")
        capped_bits = bits[self.interval[0] : self.interval[1]]
        return capped_bits

    def get_output_length(self):
        return self.output_length


@dataclass
class GenerationContext:
    client: Any
    history: str
    system_prompt: str
    max_length: int = 500
    temperature: float = 1.0


class MajorityVoteHash(HashFunction):
    def __init__(
        self,
        pca_dir: str,
        n_samples: int = 15,
        n_components: int = 0,
    ):
        super().__init__()
        self.components = np.load(
            f"{pca_dir}/pca_components.npy"
        )  # (embed_dim, n_components)
        self.mean = np.load(f"{pca_dir}/pca_mean.npy")  # (n_components, )
        self.thresholds = np.load(f"{pca_dir}/pca_thresholds.npy")  # (n_components, )
        self.n_components = n_components or self.components.shape[0]
        self.n_samples = n_samples
        self.output_length = 1

    def calibrate(self, ctx: GenerationContext) -> tuple[int]:
        """
        Call each time before hashing to calibrate for the context
        """
        hashes: list[np.ndarray] = []
        hash_counts: dict[tuple[int], int] = dict()
        for _ in range(self.n_samples):
            response = generate_response(
                client=ctx.client,
                prompt=ctx.history,
                system_prompt=ctx.system_prompt,
                max_length=ctx.max_length,
                temperature=ctx.temperature,
            ).strip()

            if response:
                emb = self._to_numpy_array(self._embed_fn(ctx.client, response))
                z = (emb - self.mean) @ self.components.T
                bits = (z > self.thresholds).astype(np.int8).ravel()
                print(f"calibrating bits:{bits}")
                hashes.append(bits)
                key = tuple(bits.tolist())
                hash_counts[key] = hash_counts.get(key, 0) + 1

        if not hashes:
            raise ValueError("Calibration failed: Error in response generation")

        majority_key = max(hash_counts, key=lambda k: hash_counts[k])
        self._majority = np.array(majority_key)
        print(self._majority)
        return majority_key

    def __call__(self, emb):
        if self._majority is None:
            raise RuntimeError("Hash not calibrated by sampling.")
        emb = self._to_numpy_array(emb)
        z = (emb - self.mean) @ self.components.T  # (n_components,)
        bits = (z > self.thresholds).astype(np.int8).ravel()
        bit = [0] if np.array_equal(bits, self._majority) else [1]

        # slice out the bits to use
        print(f"actual bits: {bits}, hashed bit: {bit}")
        return np.array(bit)

    def _embed_fn(self, client: OpenAI, text: str):
        embedding = (
            client.embeddings.create(
                input=[text],
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )
        return embedding

    def get_output_length(self):
        return 1


class OracleHash(HashFunction):
    def __init__(
        self, output_length: int, error_rate: float = 0.0, seed: Optional[int] = None
    ):
        super().__init__()
        self.output_length = output_length
        self.error_rate = error_rate
        self.hash_memory: Dict[bytes, np.ndarray] = {}
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, emb, corrupt: bool = False) -> np.ndarray:
        # Use embedding as key (convert to string for dict key)
        emb_array = self._to_numpy_array(emb)
        key = emb_array.tobytes()

        # If we haven't seen this embedding before, generate random bits
        if key not in self.hash_memory:
            self.hash_memory[key] = np.random.randint(0, 2, self.output_length)

        bits = self.hash_memory[key]

        # Apply corruption during retrieval if requested
        if corrupt and self.error_rate > 0:
            mask = np.random.random(bits.shape) < self.error_rate
            bits = np.logical_xor(bits, mask).astype(int)
        return bits

    def get_output_length(self):
        return self.output_length
