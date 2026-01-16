"""
Embeddings-based steganography system for hiding messages in text
"""

# Core system
# System Prompts
from .config.system_prompts import (
    CORPORATE_MONOLOGUE,
    CORPORATE_MONOLOGUE_ALT,
    ONE_WAY_MONOLOGUE,
    TWO_WAY_DYNAMIC,
)

# Encoders
from .core.encoder import (
    CharacterEncoder,  # Abstract base class
    CiphertextEncoder,
    Encoder,  # Abstract base class
    MinimalEncoder,
    StandardEncoder,
)

# Error correction
from .core.error_correction import (
    ConvolutionalCode,
    ErrorCorrection,  # Abstract base class
    RepetitionCode,
)

# Hash functions
from .core.hash_functions import (
    HashFunction,  # Abstract base class
    MajorityVoteHash,
    OracleHash,
    PCAHash,
    RandomProjectionHash,
)

# Simulation
from .core.simulation import Simulator
from .core.steg_system import (
    SentenceStegSystem,
    StegSystem,
    StoryStegSystem,
)
from .core.summary_system import SummarySystem
from .core.new_unit_test_system import UnitTestSystem

__version__ = "0.1.0"

__all__ = [
    # Core
    "StegSystem",
    "SentenceStegSystem",
    "StoryStegSystem",
    "SummarySystem",
    "UnitTestSystem",
    # Encoders
    "Encoder",  # Include base class for extension
    "CharacterEncoder",  # Include base class for extension
    "StandardEncoder",
    "MinimalEncoder",
    "CiphertextEncoder",
    # Error correction
    "ErrorCorrection",  # Include base class for extension
    "RepetitionCode",
    "ConvolutionalCode",
    # Hash functions
    "HashFunction",  # Include base class for extension
    "RandomProjectionHash",
    "PCAHash",
    "OracleHash",
    "MajorityVoteHash",
    # Simulation
    "Simulator",
    # System Prompts
    "CORPORATE_MONOLOGUE",
    "CORPORATE_MONOLOGUE_ALT",
    "TWO_WAY_DYNAMIC",
    "ONE_WAY_MONOLOGUE",
]
