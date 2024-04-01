# Import this file first when testing to make entire program deterministic. It sets everything besides the PRF, but for now its IV is fixed and the key is fixed in the testing file.

# config_determinism.py
import random
import numpy as np
import torch

def set_deterministic_behavior():
    # Set seed for Python's built-in random module
    random.seed(0)

    # Set seed for numpy
    np.random.seed(0)

    # Set seed and deterministic options for PyTorch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Any other deterministic settings can be added here
