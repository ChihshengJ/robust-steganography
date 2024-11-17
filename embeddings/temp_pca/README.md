"""
Embeddings-based steganography system for hiding messages in text
"""

# Core system
from embeddings.src.core.steg_system import StegSystem

# Encoders
from embeddings.src.core.encoder import (
    Encoder,
    CharacterEncoder, 
    StandardEncoder
)

# Error correction
from embeddings.src.core.error_correction import (
    ErrorCorrection,
    RepetitionCode,
    ConvolutionalCode
)

# Hash functions
from embeddings.src.core.hash_functions import (
    HashFunction,
    RandomProjectionHash,
    PCAHash,
    OracleHash
)

# Simulation
from embeddings.src.core.simulation import Simulator

__version__ = "0.1.0"

__all__ = [
    # Core
    'StegSystem',
    
    # Encoders
    'Encoder',
    'CharacterEncoder',
    'StandardEncoder',
    
    # Error correction
    'ErrorCorrection', 
    'RepetitionCode',
    'ConvolutionalCode',
    
    # Hash functions
    'HashFunction',
    'RandomProjectionHash', 
    'PCAHash',
    'OracleHash',
    
    # Simulation
    'Simulator'
]