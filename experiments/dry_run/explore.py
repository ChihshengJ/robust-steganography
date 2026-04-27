import json
import os
from pathlib import Path

DRY_RUN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DRY_RUN_DIR / "results"

SYSTEMS = ["topicqa", "story", "litreview"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_texts(system: str) -> dict[str, list[dict]]:
    """Load texts grouped by type: stego, cover_c1, cover_c2."""
    records = read_jsonl(RESULTS_DIR / f"{system}_texts.jsonl")
    grouped: dict[str, list[dict]] = {"stego": [], "cover_c1": [], "cover_c2": []}
    for r in records:
        tt = r["text_type"]
        if tt in grouped:
            grouped[tt].append(r)
    # Sort by prompt_idx for consistent pairing
    for tt in grouped:
        grouped[tt].sort(key=lambda r: r["prompt_idx"])
    return grouped


def main():
    texts = load_texts("litreview")
    print(texts["stego"][3])
    print(texts["cover_c2"][3])


if __name__ == "__main__":
    main()
