import math
import random
from collections import Counter
from typing import List, Tuple
from scipy.stats import norm

def get_keys_to_use(m: List[int], keys: List[bytes]) -> List[bytes]:
    """Get keys corresponding to 1s in message."""
    return [key for i, key in enumerate(keys) if m[i] == 1]

def sample_key(keys: List[bytes]) -> bytes:
    """Randomly sample a key from the list."""
    return random.choice(keys)

def detect(T: int, s_g: int, n: int, epsilon: float) -> bool:
    """
    Detect if a particular watermark bit is present (1) or not (0) 
    in a multi-bit scenario (n bits total), ensuring overall error ≤ epsilon.

    We use a one-sided z-test against 0.5, with a union-bound-based threshold
    z = Phi^{-1}(1 - epsilon/n).

    Parameters
    ----------
    T : int
        Total number of tokens.
    s_g : int
        Number of tokens labeled '1' under this bit's PRF.
    n : int
        Total bits in the hidden message.
    epsilon : float
        Desired overall error probability across all n bits.

    Returns
    -------
    bool
        True => detect bit=1, False => bit=0.

    Explanation
    -----------
    1. alpha_bit = epsilon / n
    2. threshold = z = Phi^{-1}(1 - alpha_bit)
    3. z_value = 2 * sqrt(T) * ((s_g/T) - 0.5)
    4. Compare z_value > threshold
    """
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")

    from scipy.stats import norm
    alpha_bit = epsilon / n
    z = norm.ppf(1.0 - alpha_bit)

    X = s_g / float(T)  # fraction
    z_value = 2.0 * math.sqrt(T) * (X - 0.5)
    return z_value > z


def generate_n_grams_with_counts(text: List[str], n: int) -> Counter:
    """Generate n-grams with their counts."""
    n_grams = []
    for i in range(len(text) - n + 1):
        n_gram = tuple(text[i:i+n])
        n_grams.append(n_gram)
    return Counter(n_grams)

def count_maintained_n_grams(text_1: List[str], text_2: List[str], n: int) -> int:
    """Count n-grams maintained between two texts."""
    n_grams_list_1 = generate_n_grams_with_counts(text_1, n)
    n_grams_list_2 = generate_n_grams_with_counts(text_2, n)
    
    total_maintained = 0
    for n_gram, count in n_grams_list_1.items():
        total_maintained += min(count, n_grams_list_2.get(n_gram, 0))
    
    return total_maintained
