import numpy as np
import torch
from scipy import stats

from ..prf import AESPRF, HMACPRF


def test_prf_bias():
    """
    Test PRF implementations for statistical bias.

    This test:
    1. Generates many PRF outputs with different keys and contexts
    2. Checks if the distribution of 1s and 0s is close to 50/50
    3. Uses chi-square test to verify uniformity
    4. Tests both AES and HMAC implementations
    """
    vocab_size = 100  # Small vocab for testing
    n_trials = 10000  # Number of different keys/contexts to test
    alpha = 0.001  # Significance level for statistical tests

    prfs = [
        ("AES PRF", AESPRF(vocab_size=vocab_size, max_token_id=vocab_size - 1)),
        ("HMAC PRF", HMACPRF(vocab_size=vocab_size, max_token_id=vocab_size - 1)),
    ]

    for prf_name, prf in prfs:
        print(f"\nTesting {prf_name} for bias...")

        # Collect statistics across multiple trials
        total_ones = 0
        total_bits = 0

        # Test with different keys, contexts, and salts
        for i in range(n_trials):
            key = prf.generate_key()
            # Create n_gram with small token IDs
            n_gram = {
                "input_ids": torch.tensor(
                    [[i % vocab_size for i in range(5)]]
                )  # Keep IDs within vocab
            }
            salt = i % 256  # Keep salt within byte range

            labels = prf(key, salt, n_gram, 5)
            total_ones += sum(labels)
            total_bits += len(labels)

        # Calculate observed proportions
        p_one = total_ones / total_bits
        p_zero = 1 - p_one

        print(f"Proportion of 1s: {p_one:.4f}")
        print(f"Proportion of 0s: {p_zero:.4f}")

        # Chi-square test for uniformity
        expected = np.array([total_bits / 2, total_bits / 2])  # Expected counts (50/50)
        observed = np.array([total_bits - total_ones, total_ones])
        chi2, p_value = stats.chisquare(observed, expected)

        print(f"Chi-square test p-value: {p_value:.4f}")

        # Test should fail if distribution is significantly non-uniform
        assert p_value > alpha, f"{prf_name} shows significant bias (p={p_value:.4f})"

        # Additional sanity checks
        assert abs(p_one - 0.5) < 0.01, f"{prf_name} deviates too far from 50/50 split"


def test_prf_independence():
    """
    Test that PRF outputs are independent across different keys/contexts.

    This test:
    1. Generates outputs with different keys but same context
    2. Generates outputs with same key but different contexts
    3. Checks for correlations between outputs
    """
    vocab_size = 100
    n_pairs = 1000

    prfs = [
        ("AES PRF", AESPRF(vocab_size=vocab_size, max_token_id=vocab_size - 1)),
        ("HMAC PRF", HMACPRF(vocab_size=vocab_size, max_token_id=vocab_size - 1)),
    ]

    for prf_name, prf in prfs:
        print(f"\nTesting {prf_name} for independence...")

        # Test different keys, same context
        correlations_keys = []
        base_n_gram = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]])  # Small token IDs
        }

        for i in range(n_pairs):
            salt = i % 256  # Keep salt within byte range
            key1 = prf.generate_key()
            key2 = prf.generate_key()

            out1 = prf(key1, salt, base_n_gram, 5)
            out2 = prf(key2, salt, base_n_gram, 5)

            correlation = np.corrcoef(out1, out2)[0, 1]
            correlations_keys.append(abs(correlation))

        # Test same key, different contexts
        correlations_contexts = []
        base_key = prf.generate_key()

        for i in range(n_pairs):
            # Keep token IDs within vocab range
            n_gram1 = {
                "input_ids": torch.tensor(
                    [[np.random.randint(0, vocab_size) for _ in range(5)]]
                )
            }
            n_gram2 = {
                "input_ids": torch.tensor(
                    [[np.random.randint(0, vocab_size) for _ in range(5)]]
                )
            }
            salt = i % 256

            out1 = prf(base_key, salt, n_gram1, 5)
            out2 = prf(base_key, salt, n_gram2, 5)

            correlation = np.corrcoef(out1, out2)[0, 1]
            correlations_contexts.append(abs(correlation))

        # Print results
        print(
            f"Mean absolute correlation (different keys): {np.mean(correlations_keys):.4f}"
        )
        print(
            f"Mean absolute correlation (different contexts): {np.mean(correlations_contexts):.4f}"
        )

        # Test for significant correlations
        assert np.mean(correlations_keys) < 0.1, (
            f"{prf_name} shows correlation between different keys"
        )
        assert np.mean(correlations_contexts) < 0.1, (
            f"{prf_name} shows correlation between different contexts"
        )


if __name__ == "__main__":
    import torch

    test_prf_bias()
    test_prf_independence()

