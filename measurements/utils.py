import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

import numpy as np
from tqdm import tqdm
from typing_extensions import Callable

from embeddings import Encoder, StegSystem


@dataclass
class ExperimentConfig:
    """All configuration for an experiment run."""

    tampering_levels: list[float]
    attack_configs: list[dict]
    system: StegSystem
    num_bits: int
    num_messages: int
    num_stego_per_message: int
    messages: list[list[int]] | None = None
    runs: int = 5
    history: list[str] | str = field(default_factory=list)
    seed: int | None = None
    checkpoint_path: Path = Path("checkpoints/exp_checkpoint.pkl")
    output_path: Path = Path("exp_results")
    save_texts: bool = False
    max_saved_examples: int = 200
    resume: bool = True
    checkpoint_after_each_stego: bool = False


@dataclass
class CheckpointState:
    """Tracks experiment progress for resumption."""

    message_index: int = 0
    current_message_stego_complete: bool = False
    stego_gen_index: int = 0
    attack_index: int = 0
    tampering_index: int = 0
    stego_index: int = 0
    run_index: int = 0
    all_ret: dict = field(default_factory=dict)
    message_results: dict = field(default_factory=dict)
    texts_saved_count: int = 0
    random_state: Any = None
    current_stego_texts: list[str] = field(default_factory=list)

    def reset_for_new_message(self):
        """Reset indices when moving to a new message."""
        self.attack_index = 0
        self.tampering_index = 0
        self.stego_index = 0
        self.run_index = 0
        self.current_message_stego_complete = False
        self.stego_gen_index = 0
        self.current_stego_texts = []
        self.message_results = {}

    def reset_for_new_attack(self):
        self.tampering_index = 0
        self.stego_index = 0
        self.run_index = 0

    def reset_for_new_tampering(self):
        self.stego_index = 0
        self.run_index = 0

    def reset_for_new_stego(self):
        self.run_index = 0


class CheckpointManager:
    """Handles saving and loading experiment checkpoints."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState):
        with open(self.path, "wb") as f:
            pickle.dump(state.__dict__, f)

    def load(self) -> CheckpointState:
        if not self.path.exists():
            return CheckpointState()
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        state = CheckpointState()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state

    def exists(self) -> bool:
        return self.path.exists()


class ProgressTracker:
    """Manages nested tqdm progress bars."""

    def __init__(self):
        self._bars: dict[str, tqdm] = {}

    def create(
        self,
        name: str,
        total: int,
        desc: str,
        position: int = 0,
        initial: int = 0,
        leave: bool = True,
    ) -> tqdm:
        bar = tqdm(
            total=total, desc=desc, position=position, initial=initial, leave=leave
        )
        self._bars[name] = bar
        return bar

    def update(self, name: str, n: int = 1):
        if name in self._bars:
            self._bars[name].update(n)

    def set_position(self, name: str, n: int):
        if name in self._bars:
            self._bars[name].n = n
            self._bars[name].refresh()

    def close(self, name: str):
        if name in self._bars:
            self._bars[name].close()
            del self._bars[name]

    def close_all(self):
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


class TextLogger:
    """Logs stego/attack/recovery examples to JSONL."""

    def __init__(self, path: Path, max_examples: int):
        self.path = path
        self.max_examples = max_examples
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        attack_label: str,
        attack_type: str,
        tampering: float,
        msg_idx: int,
        stego_idx: int,
        run_idx: int,
        message: list[int],
        stego_text: str,
        attacked_text: str,
        recovered: list[int],
    ) -> bool:
        """Log an example. Returns True if logged, False if limit reached."""
        if self.count >= self.max_examples:
            return False

        record = {
            "timestamp": time.time(),
            "attack_label": attack_label,
            "attack_type": attack_type,
            "tampering": tampering,
            "message_index": msg_idx,
            "stego_index": stego_idx,
            "run_index": run_idx,
            "message_bits": message,
            "stego_texts": stego_text,
            "attacked_texts": attacked_text,
            "recovered": recovered,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self.count += 1
        return True


class BypassEncoder(Encoder):
    """Encoder that passes data through unchanged."""

    def __init__(self):
        pass

    @override
    def encode(self, data: list[int]) -> list[int]:
        return data

    @override
    def decode(self, bits: list[int]) -> list[int]:
        return bits


def index_reducer(index: int) -> Callable[[np.ndarray], np.ndarray]:
    def reducer(bits: np.ndarray) -> np.ndarray:
        return np.array([bits[index]], dtype=np.int8)

    return reducer


def threshold_sum_reducer(
    threshold: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    def reducer(bits: np.ndarray) -> np.ndarray:
        t = threshold if threshold is not None else len(bits) / 2
        return np.array([1 if np.sum(bits) > t else 0], dtype=np.int8)

    return reducer


def slice_reducer(start: int, end: int) -> Callable[[np.ndarray], np.ndarray]:
    """Returns bits[start:end] - preserves original multi-bit behavior."""

    def reducer(bits: np.ndarray) -> np.ndarray:
        return bits[start:end]

    return reducer
