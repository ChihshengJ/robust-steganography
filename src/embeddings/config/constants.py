from dataclasses import dataclass

STEGO_GEN_MODEL = "gpt-4.1-mini"


@dataclass
class BacktrackConfig:
    max_attempts_per_step: int = 30
    max_backtracks: int = 5
    collect_alternatives: bool = True
    max_alternatives: int = 3
