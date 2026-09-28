"""Phase 2a: Token / word counts of the detection set, and its length matching.

Reads every (stegotext, normal generation) pair of the detection cells
(``stegoanalysis_common.load_detection_set``) and emits:

    data/experiments/phase2_metrics/detection/token_counts.jsonl
    data/experiments/phase2_metrics/detection/token_counts.csv
    data/experiments/phase2_metrics/detection/token_counts_summary.json

For every text it writes ``{token_count, word_count, char_count}`` and, for
stegotexts, ``bits_per_token = F / token_count``. The summary gives, per cell,
the stats of each class and of the pair length ratio (normal words / stego
words): a length gap alone separates the classes, so it should stay near 1.

Token counts use ``tiktoken`` (``o200k_base``).

Usage:
    python -m experiments.phase2_metrics.phase2a_token_counts
    python -m experiments.phase2_metrics.phase2a_token_counts --systems story
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.utils.stegoanalysis_common import (
    add_common_args,
    detection_metrics_dir,
    load_detection_set,
    write_jsonl,
)
from experiments.utils.token_counter import bits_per_token, count_tokens, count_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FIELDS = [
    "uid",
    "system",
    "cell",
    "config",
    "F",
    "text_type",
    "prompt_idx",
    "token_count",
    "word_count",
    "char_count",
    "bits_per_token",
]


def _build_record(item: dict) -> dict:
    text = item["text"]
    tok = count_tokens(text)
    stego = item["label"] == 1
    return {
        "uid": item["uid"],
        "system": item["system"],
        "cell": item["cell"],
        "config": item["config"],
        "F": item["F"],
        "text_type": "stego" if stego else "normal",
        "prompt_idx": item["prompt_idx"],
        "token_count": tok,
        "word_count": count_words(text),
        "char_count": len(text),
        "bits_per_token": bits_per_token(item["F"], tok) if stego else None,
    }


def _stat_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _summarize(records: list[dict]) -> dict:
    """{system: {cell: {stego: ..., normal: ..., length_ratio: ...}}}."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["system"], r["cell"])].append(r)

    summary: dict[str, dict] = {}
    for (system, cell), group in sorted(by_cell.items()):
        block: dict = {"config": group[0]["config"], "F": group[0]["F"]}
        for text_type in ("stego", "normal"):
            g = [r for r in group if r["text_type"] == text_type]
            block[text_type] = {
                "n": len(g),
                "token_count": _stat_block([r["token_count"] for r in g]),
                "word_count": _stat_block([r["word_count"] for r in g]),
            }
        block["stego"]["bits_per_token"] = _stat_block(
            [r["bits_per_token"] for r in group if r["bits_per_token"] is not None]
        )
        words = {(r["prompt_idx"], r["text_type"]): r["word_count"] for r in group}
        block["length_ratio"] = _stat_block(
            [
                words[(p, "normal")] / words[(p, "stego")]
                for p, t in words
                if t == "stego" and (p, "normal") in words and words[(p, "stego")]
            ]
        )
        summary.setdefault(system, {})[cell] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2a: token / word counts")
    add_common_args(parser)
    args = parser.parse_args()

    out_dir = detection_metrics_dir(args.data_dir)
    records: list[dict] = []
    for system in args.systems:
        items = load_detection_set(args.data_dir, system, args.dirs)
        if not items:
            log.warning("[%s] no detection pairs found", system)
            continue
        log.info("[%s] %d texts", system, len(items))
        records.extend(_build_record(it) for it in items)

    write_jsonl(out_dir / "token_counts.jsonl", records)
    with open(out_dir / "token_counts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    summary = _summarize(records)
    with open(out_dir / "token_counts_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for system, cells in summary.items():
        for cell, block in cells.items():
            lr = block["length_ratio"]
            log.info(
                "[%s] %s: %d pairs, mean normal/stego words %.3f",
                system,
                cell,
                lr.get("n", 0),
                lr.get("mean", float("nan")),
            )
    log.info("Wrote %d records to %s", len(records), out_dir)


if __name__ == "__main__":
    main()
