"""JSONL I/O utilities with checkpoint support."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file with atomic flush."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file. Returns empty list if file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def inputs_path(phase1_dir: str | Path, system: str) -> Path:
    """A Phase 1 dir's selected inputs, written by select_inputs."""
    return Path(phase1_dir) / f"{system}_inputs.json"


def read_stego_records(phase1_dir: str | Path, system: str) -> list[dict]:
    """A Phase 1 dir's stego records, restricted to its selected inputs when
    ``{system}_inputs.json`` exists (the payload grid's shared inputs, see
    ``experiments.phase1_generation.select_inputs``); all of them otherwise."""
    records = read_jsonl(Path(phase1_dir) / f"{system}_stego.jsonl")
    path = inputs_path(phase1_dir, system)
    if not path.exists():
        return records
    selected = set(json.loads(path.read_text())["prompt_idx"])
    return [r for r in records if r.get("prompt_idx") in selected]


def load_completed_ids(path: str | Path) -> set[str]:
    """Scan a JSONL file and return a set of record ids for checkpoint resumption."""
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("id")
                if rid:
                    ids.add(rid)
            except json.JSONDecodeError:
                continue
    return ids


def load_records_map(path: str | Path) -> dict[str, dict]:
    """Load all records from a JSONL file, keyed by id."""
    records = read_jsonl(path)
    return {r["id"]: r for r in records if "id" in r}


_TYPE_SHORT = {"stego": "s", "normal": "n", "cover_c1": "c1", "cover_c2": "c2"}


def make_record_id(system: str, text_type: str, prompt_idx: int) -> str:
    """Create a deterministic composite ID matching the experiment.md schema.

    Example: make_record_id("story", "stego", 42) -> "story_s_042"
    """
    short = _TYPE_SHORT.get(text_type, text_type)
    return f"{system}_{short}_{prompt_idx:03d}"


def model_slug(model: str) -> str:
    """Short, id-safe model name, e.g. 'deepseek-ai/DeepSeek-V4-Flash' ->
    'deepseek-v4-flash'. Used in record ids and output subdir names."""
    name = model.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-")


def stable_seed(key: str) -> int:
    """Deterministic seed in [0, 2**31) derived from ``key``.

    Uses SHA-256 rather than the builtin ``hash()``, which is salted per
    process (PYTHONHASHSEED) and so gives a different seed on every run.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
