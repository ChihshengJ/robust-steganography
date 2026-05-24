from dataclasses import dataclass

STEGO_GEN_MODEL = "gpt-4o-mini"


@dataclass
class BacktrackConfig:
    max_attempts_per_step: int = 20
    max_backtracks: int = 5
    collect_alternatives: bool = True
    max_alternatives: int = 2
