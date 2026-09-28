"""Attacker-model bookkeeping shared by Phase 3 (attacks) and Phase 4 (decoding,
metrics, tables).

Every attacked record carries ``attacker_model``: the LLM that paraphrased or
translated it, or None for attacks that use no LLM (synonym). Aggregations
group by it so results from different attackers never pool into one cell.
"""

from __future__ import annotations

# The attacker every LLM attack used before the attacker became configurable.
# Records without an ``attacker_model`` field ran on it; ids of its records
# carry no attacker suffix.
DEFAULT_ATTACKER = "gpt-4.1"

# The attacker for new runs: a family used for neither G nor the writers.
ATTACKER_MODEL = "z-ai/glm-5.3-flash"
ATTACKER_PROVIDER = "openrouter"

# Phase 3 attack_type values that call an LLM.
LLM_ATTACK_TYPES = frozenset({"paraphrase", "translate"})


def attacker_of(record: dict) -> str | None:
    """The LLM that produced an attacked (or decoded) record, or None.

    Records written before attacker models were configurable have no
    ``attacker_model`` field; every LLM attack then ran on DEFAULT_ATTACKER.
    """
    if "attacker_model" in record:
        return record["attacker_model"]
    if record.get("attack_type") in LLM_ATTACK_TYPES:
        return DEFAULT_ATTACKER
    return None

