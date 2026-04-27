"""JSONL I/O utilities with checkpoint support."""

from __future__ import annotations

import json
import os
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


_TYPE_SHORT = {"stego": "s", "cover_c1": "c1", "cover_c2": "c2"}


def make_record_id(system: str, text_type: str, prompt_idx: int) -> str:
    """Create a deterministic composite ID matching the experiment.md schema.

    Example: make_record_id("topicqa", "stego", 42) -> "topicqa_s_042"
    """
    short = _TYPE_SHORT.get(text_type, text_type)
    return f"{system}_{short}_{prompt_idx:03d}"
