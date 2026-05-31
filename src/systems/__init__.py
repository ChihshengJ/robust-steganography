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

# Simulation
from .core.embeddings.simulation import Simulator

# Encoders
from .core.encoder import (
    BypassEncoder,
    CharacterEncoder,
    CiphertextEncoder,
    Encoder,
    MinimalEncoder,
    StandardEncoder,
)

# Error correction
from .core.error_correction import (
    ConvolutionalCode,
    ErrorCorrection,
    RepetitionCode,
)

# Hash functions
from .core.hash_functions import (
    BitsPerGroupStub,
    HashFunction,
    MajorityVoteHash,
    OracleHash,
    PCAHash,
    RandomProjectionHash,
)
from .core.litreview import LitReviewSystem
from .core.steg_system import (
    OracleStegSystem,
    SentenceStegSystem,
    StegSystem,
)
from .core.story_gen import StorySystem
from .core.topicqa import TopicQASystem

__version__ = "0.1.0"

__all__ = [
    # Core
    "StegSystem",
    "OracleStegSystem",
    "SentenceStegSystem",
    "StorySystem",
    "LitReviewSystem",
    "TopicQASystem",
    # Encoders
    "Encoder",
    "BypassEncoder",
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
    "BitsPerGroupStub",
    # Simulation
    "Simulator",
    # System Prompts
    "CORPORATE_MONOLOGUE",
    "CORPORATE_MONOLOGUE_ALT",
    "TWO_WAY_DYNAMIC",
    "ONE_WAY_MONOLOGUE",
]
