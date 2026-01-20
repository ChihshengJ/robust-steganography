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
    Encoder,  # Abstract base class
    CharacterEncoder,
    CiphertextEncoder,
    MinimalEncoder,
    StandardEncoder,
)

# Error correction
from .core.error_correction import (
    ErrorCorrection,  # Abstract base class
    ConvolutionalCode,
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
    StegSystem, # Abstract base class
    OracleStegSystem,
    SentenceStegSystem,
)
from .core.story_system import StoryStegSystem
from .core.summary_system import SummarySystem
from .core.new_unit_test_system import UnitTestSystem

__version__ = "0.1.0"

__all__ = [
    # Core
    "StegSystem",
    "OracleStegSystem",
    "SentenceStegSystem",
    "StoryStegSystem",
    "SummarySystem",
    "UnitTestSystem",
    # Encoders
    "Encoder",
    "CharacterEncoder",
    "StandardEncoder",
    "MinimalEncoder",
    "CiphertextEncoder",
    # Error correction
    "ErrorCorrection",
    "RepetitionCode",
    "ConvolutionalCode",
    # Hash functions
    "HashFunction",
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
