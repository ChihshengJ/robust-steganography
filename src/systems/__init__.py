from .env import load_env as _load_env

_load_env()


# System Prompts
from .config.system_prompts import (
    CORPORATE_MONOLOGUE,
    CORPORATE_MONOLOGUE_ALT,
    ONE_WAY_MONOLOGUE,
    TWO_WAY_DYNAMIC,
)
from .core.embeddings.embedding_steg_system import (
    EmbeddingStegSystem,
    OracleStegSystem,
    SentenceStegSystem,
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
from .core.baselines import DiscopLM, DiscopSystem, MeteorLM, MeteorSystem
from .core.litreview import LitReviewSystem
from .core.steg_system import StegSystem
from .core.story_gen import StorySystem
from .core.topicqa import TopicQASystem

__version__ = "0.1.0"

__all__ = [
    # Core
    "StegSystem",
    "EmbeddingStegSystem",
    "OracleStegSystem",
    "SentenceStegSystem",
    "StorySystem",
    "LitReviewSystem",
    "TopicQASystem",
    # In-house token-level baselines (Meteor, Discop)
    "MeteorSystem",
    "MeteorLM",
    "DiscopSystem",
    "DiscopLM",
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
