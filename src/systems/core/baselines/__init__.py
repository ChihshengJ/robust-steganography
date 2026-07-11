"""In-house token-level steganography baselines (Meteor, Discop).

These are vendored, adapted third-party schemes used only for baseline
comparison in the experiments; they are deliberately isolated from the paper's
own systems under ``systems.core``. Both drive a local HuggingFace GPT-2 and
implement the standard :class:`StegSystem` ``hide_message``/``recover_message``
API, so the existing phase pipeline can treat them like any other system.
"""

from ._discop_backend import DiscopLM
from ._meteor_backend import MeteorLM
from .discop_system import DiscopSystem
from .meteor_system import MeteorSystem

__all__ = ["MeteorSystem", "MeteorLM", "DiscopSystem", "DiscopLM"]
