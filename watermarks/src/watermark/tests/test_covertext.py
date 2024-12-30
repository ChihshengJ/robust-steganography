import pytest
import numpy as np
from watermark import SmoothCovertextCalculator

def test_smooth_calculator_basic():
    """Test basic functionality of SmoothCovertextCalculator."""
    calculator = SmoothCovertextCalculator()
    
    # Test with typical values
    n = 3  # 3-bit message
    epsilon = 0.05  # 95% success probability
    delta = 0.1  # typical perturbation strength
    
    length = calculator.get_covertext_length(n, epsilon, delta)
    
    # Basic sanity checks
    assert isinstance(length, int)
    assert length > 0
    assert length >= n  # Length must be at least number of bits

def test_smooth_calculator_scaling():
    """Test that length scales properly with parameters."""
    calculator = SmoothCovertextCalculator()
    
    # Base case
    base_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.1
    )
    
    # Double message length should roughly double required length
    double_msg_length = calculator.get_covertext_length(
        n=2,
        epsilon=0.05,
        delta=0.1
    )
    assert double_msg_length > base_length
    
    # Higher confidence (smaller epsilon) should increase length
    high_conf_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.01,  # 99% vs 95%
        delta=0.1
    )
    assert high_conf_length > base_length
    
    # Smaller delta should require longer text
    small_delta_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.05  # Half the perturbation
    )
    assert small_delta_length > base_length

def test_smooth_calculator_edge_cases():
    """Test edge cases and invalid inputs."""
    calculator = SmoothCovertextCalculator()
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        calculator.get_covertext_length(n=0, epsilon=0.05, delta=0.1)
    
    with pytest.raises(ValueError):
        calculator.get_covertext_length(n=1, epsilon=0, delta=0.1)
    
    with pytest.raises(ValueError):
        calculator.get_covertext_length(n=1, epsilon=0.05, delta=0)
    
    with pytest.raises(ValueError):
        calculator.get_covertext_length(n=1, epsilon=1.1, delta=0.1)
    
    # Test extreme but valid values
    base_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.1
    )
    
    tiny_epsilon = calculator.get_covertext_length(
        n=1,
        epsilon=0.001,  # 99.9% confidence
        delta=0.1
    )
    assert tiny_epsilon > base_length
    
    tiny_delta = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.01  # Very small perturbation
    )
    assert tiny_delta > base_length

def test_smooth_calculator_reproducibility():
    """Test that results are deterministic."""
    calculator = SmoothCovertextCalculator()
    
    params = dict(n=3, epsilon=0.05, delta=0.1)
    
    length1 = calculator.get_covertext_length(**params)
    length2 = calculator.get_covertext_length(**params)
    
    assert length1 == length2

def test_smooth_calculator_custom_p0():
    """Test with non-default p0 values."""
    calculator = SmoothCovertextCalculator()
    
    # Test with different p0 values
    base_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.1,
        p0=0.5  # Default
    )
    
    skewed_length = calculator.get_covertext_length(
        n=1,
        epsilon=0.05,
        delta=0.1,
        p0=0.7  # Skewed probability
    )
    
    # Skewed probability should require different length
    assert skewed_length != base_length 

def test_print_practical_lengths():
    """Print covertext lengths for various practical scenarios."""
    calculator = SmoothCovertextCalculator()
    
    print("\nRequired covertext lengths for different scenarios:")
    print("------------------------------------------------")
    
    # Standard case
    length = calculator.get_covertext_length(
        n=3,  # 3-bit message
        epsilon=0.05,  # 95% confidence
        delta=0.1  # typical perturbation
    )
    print(f"Standard case (3 bits, 95% confidence, δ=0.1): {length} tokens")
    
    # High confidence case
    length = calculator.get_covertext_length(
        n=3,
        epsilon=0.01,  # 99% confidence
        delta=0.1
    )
    print(f"High confidence case (3 bits, 99% confidence, δ=0.1): {length} tokens")
    
    # Long message case
    length = calculator.get_covertext_length(
        n=8,  # 1 byte
        epsilon=0.05,
        delta=0.1
    )
    print(f"Long message case (8 bits, 95% confidence, δ=0.1): {length} tokens")
    
    # Subtle watermark case
    length = calculator.get_covertext_length(
        n=3,
        epsilon=0.05,
        delta=0.05  # smaller perturbation
    )
    print(f"Subtle watermark case (3 bits, 95% confidence, δ=0.05): {length} tokens")
    
    # Very reliable case
    length = calculator.get_covertext_length(
        n=3,
        epsilon=0.001,  # 99.9% confidence
        delta=0.1
    )
    print(f"Very reliable case (3 bits, 99.9% confidence, δ=0.1): {length} tokens")
    
    # Short story case
    length = calculator.get_covertext_length(
        n=32,  # 4 bytes
        epsilon=0.05,
        delta=0.1
    )
    print(f"Short story case (32 bits, 95% confidence, δ=0.1): {length} tokens")
    
    print("------------------------------------------------\n")
    
    # This test always passes - it's just for printing info
    assert True 