"""Select the payload grid's shared inputs: the first N that encoded in every cell.

Phase 1 skips an input it cannot encode (an SG premise G gives too few slots,
an LR paper whose greedy walk fails; see ``{system}_failures.jsonl``), so cells
can lose different inputs. Every cell of a task must use the same N inputs,
across F and across configurations, so an input that failed in any cell is
dropped everywhere and the next spare takes its place.

Run after Phase 1 has written every cell, with more inputs than N (``--limit``).
Writes ``{system}_inputs.json`` into each cell dir; the readers of Phase 1
records (normal generations, attacks, results) keep only those inputs.

Usage:
    # Every cell of the task, across both tracks:
    python -m experiments.phase1_generation.select_inputs --system story \\
        --dirs 'detection/story_cap*' 'recovery/story_cap*'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1_generation.phase1_normal import find_stego_dirs
from experiments.utils.io import inputs_path, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def encoded_inputs(stego_dir: Path, system: str) -> set[int]:
    """prompt_idx of every stegotext the cell holds."""
    return {
        r["prompt_idx"]
        for r in read_jsonl(stego_dir / f"{system}_stego.jsonl")
        if r.get("prompt_idx") is not None
    }


def failed_inputs(stego_dir: Path, system: str) -> list[int]:
    path = stego_dir / f"{system}_failures.jsonl"
    return sorted({r["prompt_idx"] for r in read_jsonl(path)}) if path.exists() else []


def select(per_dir: dict[Path, set[int]], n: int) -> list[int]:
    """The first n inputs, by prompt_idx, present in every cell."""
    common = set.intersection(*per_dir.values())
    return sorted(common)[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True, choices=["story", "litreview"])
    ap.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Glob patterns under phase1_texts/ selecting every cell of the grid.",
    )
    ap.add_argument("--n", type=int, default=30, help="Inputs per cell (default 30).")
    ap.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    ap.add_argument(
        "--dry-run", action="store_true", help="Report the selection; write nothing."
    )
    args = ap.parse_args()

    root = args.data_dir / "phase1_texts"
    dirs = find_stego_dirs(root, args.system, args.dirs)
    if not dirs:
        sys.exit(f"no {args.system}_stego.jsonl under {root}/{args.dirs}")

    per_dir = {d: encoded_inputs(d, args.system) for d in dirs}
    for d, present in per_dir.items():
        log.info(
            "%s: %d encoded, failed %s",
            d.relative_to(root),
            len(present),
            failed_inputs(d, args.system) or "none",
        )
    selected = select(per_dir, args.n)
    dropped = sorted(set.union(*per_dir.values()) - set(selected))
    dropped = [i for i in dropped if selected and i < selected[-1]]
    log.info(
        "selected %d inputs; dropped before the last one: %s", len(selected), dropped
    )
    if len(selected) < args.n:
        sys.exit(
            f"only {len(selected)} inputs encoded in all {len(dirs)} cells, need "
            f"{args.n}. Rerun Phase 1 on these cells with a higher --limit (it "
            "resumes), then select again."
        )
    if args.dry_run:
        return

    selection = {
        "system": args.system,
        "prompt_idx": selected,
        "n": args.n,
        "dropped": dropped,
        "cells": [str(d.relative_to(root)) for d in dirs],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for d in dirs:
        inputs_path(d, args.system).write_text(json.dumps(selection, indent=2))
    log.info("wrote %s to %d cells", inputs_path(".", args.system).name, len(dirs))


if __name__ == "__main__":
    main()
