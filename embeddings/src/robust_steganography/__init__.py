"""
Embeddings-based steganography system for hiding messages in text
"""

# Core system
from .core.steg_system import StegSystem

# Encoders
from .core.encoder import (
    Encoder,  # Abstract base class
    CharacterEncoder, # Abstract base class
    StandardEncoder,
    MinimalEncoder,
    CiphertextEncoder
)

# Error correction
from .core.error_correction import (
    ErrorCorrection,  # Abstract base class
    RepetitionCode,
    ConvolutionalCode
)

# Hash functions
from .core.hash_functions import (
    HashFunction,  # Abstract base class
    RandomProjectionHash,
    PCAHash,
    OracleHash
)

# Simulation
from .core.simulation import Simulator

__version__ = "0.1.0"

__all__ = [
    # Core
    'StegSystem',
    
    # Encoders
    'Encoder',  # Include base class for extension
    'CharacterEncoder', # Include base class for extension
    'StandardEncoder',
    'MinimalEncoder',
    'CiphertextEncoder',
    
    # Error correction
    'ErrorCorrection',  # Include base class for extension
    'RepetitionCode',
    'ConvolutionalCode',
    
    # Hash functions
    'HashFunction',  # Include base class for extension
    'RandomProjectionHash', 
    'PCAHash',
    'OracleHash',
    
    # Simulation
    'Simulator'
]