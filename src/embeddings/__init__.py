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
    CharacterEncoder,
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
from .core.litreview_system import LitReviewSystem
from .core.litreview_v2 import LitReviewSystemV2
from .core.new_unit_test_system import UnitTestSystem

# Simulation
from .core.simulation import Simulator
from .core.steg_system import (
    OracleStegSystem,
    SentenceStegSystem,
    StegSystem,  # Abstract base class
)
from .core.story_system import StoryStegSystem
from .core.summary_system import SummarySystem

__version__ = "0.1.0"

__all__ = [
    # Core
    "StegSystem",
    "OracleStegSystem",
    "SentenceStegSystem",
    "StoryStegSystem",
    "SummarySystem",
    "LitReviewSystem",
    "LitReviewSystemV2",
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
