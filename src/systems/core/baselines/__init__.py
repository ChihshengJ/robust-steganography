"""In-house token-level steganography baseline (Discop).

Discop (Ding et al., IEEE S&P 2023) is a vendored, adapted third-party scheme
used only for baseline comparison in the experiments; it is deliberately
isolated from the paper's own systems under ``systems.core``. It drives a local
HuggingFace GPT-2 and implements the standard :class:`StegSystem`
``hide_message``/``recover_message`` API, so the existing phase pipeline can
treat it like any other system.
"""

from ._discop_backend import DiscopLM
from .discop_system import DiscopSystem

__all__ = ["DiscopSystem", "DiscopLM"]
