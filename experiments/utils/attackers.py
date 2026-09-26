"""Attacker-model bookkeeping shared by Phase 3 (attacks) and Phase 4 (decoding,
metrics, tables).

Every attacked record carries ``attacker_model``: the LLM that paraphrased or
translated it, or None for attacks that use no LLM (synonym). Aggregations
group by it so results from different attackers never pool into one cell.
"""

from __future__ import annotations

import re

# The attacker every LLM attack used before the attacker became configurable.
DEFAULT_ATTACKER = "gpt-4.1"

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


def attacker_slug(model: str) -> str:
    """Short, id-safe attacker name, e.g. 'deepseek-ai/DeepSeek-V4-Flash' ->
    'deepseek-v4-flash'."""
    name = model.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-")

