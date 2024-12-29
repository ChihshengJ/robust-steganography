import math
import random
from collections import Counter
from typing import List, Tuple

def get_keys_to_use(m: List[int], keys: List[bytes]) -> List[bytes]:
    """Get keys corresponding to 1s in message."""
    return [key for i, key in enumerate(keys) if m[i] == 1]

def get_limit(num_watermarks: int) -> int:
    """Calculate appropriate length for watermarking."""
    # TODO: Implement proper calculation based on:
    # 1. Number of watermarks
    # 2. Statistical analysis
    # 3. Minimum coherence length
    # 4. Training data analysis
    return None

def sample_key(keys: List[bytes]) -> bytes:
    """Randomly sample a key from the list."""
    return random.choice(keys)

def detect(T: int, s_g: int, threshold: float = 2.0) -> bool:
    """
    Detect if watermark is present based on statistical test.
    
    Args:
        T: Total number of tokens
        s_g: Number of green list tokens
        threshold: Z-score threshold for detection
        
    Returns:
        True if watermark detected, False otherwise
    """
    z = get_z_score(T, s_g)
    return z > threshold

def get_z_score(T: int, s_g: int) -> float:
    """
    Calculate Z-score for watermark detection.
    
    Args:
        T: Total number of tokens
        s_g: Number of green list tokens
        
    Returns:
        Z-score value
    """
    # Expected value and variance under null hypothesis
    E_s = T/2
    Var_s = T/4
    
    # Z-score calculation
    if Var_s == 0:
        return 0
    return (s_g - E_s) / math.sqrt(Var_s)

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
