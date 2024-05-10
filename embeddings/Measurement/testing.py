import numpy as np
import random
from Helpers import find_cutoff_helper, find_cutoff

# Test cases
# test_data = [
#     ([1.1, 2.2, 3.3], [5.5, 6.6, 7.7], 0.1),  # No overlap
#     ([3.3, 4.4, 5.5], [5.0, 6.0, 7.0], 0.3),  # Overlap with achievable error rate
#     ([3.3, 4.4, 5.5], [5.0, 6.0, 7.0], 0.7),  # High error rate requirement
#     ([1.5, 2.5, 3.5], [1.8, 2.8, 3.8], 0.1),  # Overlap with unachievable error rate
#     ([], [1.2, 1.3], 0.1),                    # Empty little list
#     ([1.2, 1.3], [], 0.1)                     # Empty big list
# ]

# Generating large test data
def generate_no_overlap_data(size):
    little = [random.uniform(1, 50) for _ in range(size)]
    big = [random.uniform(51, 100) for _ in range(size)]
    return little, big

def generate_overlap_data(size):
    little = [random.uniform(1, 60) for _ in range(size)]
    big = [random.uniform(50, 100) for _ in range(size)]
    return little, big

# Test cases
test_data = [
    (generate_no_overlap_data(300), 0.1),  # No overlap
    (generate_overlap_data(300), 0.2),     # Overlap with moderate error rate
    (generate_overlap_data(300), 0.05)     # High overlap with low error rate
]

for i, ((little, big), error_rate) in enumerate(test_data):
    result = find_cutoff_helper(little, big, error_rate)
    print(f"Test Case {i+1}: Result = {result}")