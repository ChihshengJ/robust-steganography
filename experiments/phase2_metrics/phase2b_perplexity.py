"""Phase 2b: Perplexity of the detection set, conditioned on the task input.

Scores every text of the detection cells (``load_detection_set``) with a
HuggingFace causal LM (gpt2-large by default), conditioned on its task input x:
x is public, so the monitor's language model sees it before the text
(``PerplexityScorer.score(text, prefix=x)``). Emits

    data/experiments/phase2_metrics/detection/perplexity.jsonl
    data/experiments/phase2_metrics/detection/perplexity_summary.json

keyed by the cell-qualified ``uid``. The perplexity-only detector (a logistic
regression on it) runs in phase2c_summary.

Checkpoint-resumable: uids already scored with the same model are skipped.

Usage:
    python -m experiments.phase2_metrics.phase2b_perplexity
    python -m experiments.phase2_metrics.phase2b_perplexity --systems story --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.utils.io import append_jsonl, read_jsonl
from experiments.utils.perplexity import PerplexityScorer
from experiments.utils.stegoanalysis_common import (
    add_common_args,
    detection_metrics_dir,
    load_detection_set,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def perplexity_path(data_dir: Path) -> Path:
    return detection_metrics_dir(data_dir) / "perplexity.jsonl"


def _stat_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def _summarize(records: list[dict]) -> dict:
    """{model: {system: {cell: {stego: stats, normal: stats}}}}."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        groups[(r["model"], r["system"], r["cell"], r["text_type"])].append(
            r["perplexity"]
        )
    out: dict = {}
    for (model, system, cell, text_type), values in sorted(groups.items()):
        out.setdefault(model, {}).setdefault(system, {}).setdefault(cell, {})[
            text_type
        ] = _stat_block(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2b: perplexity scoring")
    add_common_args(parser)
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap texts per system (smoke tests).",
    )
    args = parser.parse_args()

    jsonl_path = perplexity_path(args.data_dir)
    done = {
        r["uid"] for r in read_jsonl(jsonl_path) if r.get("model") == args.model
    }
    log.info("Found %d texts already scored with %s", len(done), args.model)

    scorer = None
    n_scored = 0
    for system in args.systems:
        items = load_detection_set(args.data_dir, system, args.dirs)
        if args.limit is not None:
            items = items[: args.limit]
        pending = [it for it in items if it["uid"] not in done]
        log.info("[%s] %d texts, %d pending", system, len(items), len(pending))
        if pending and scorer is None:
            scorer = PerplexityScorer(model_name=args.model, device=args.device)
            log.info("Scorer %s ready on %s", args.model, scorer.device)
        for i, it in enumerate(pending):
            score = scorer.score(it["text"], prefix=it["input"])
            append_jsonl(
                jsonl_path,
                {
                    "uid": it["uid"],
                    "system": system,
                    "cell": it["cell"],
                    "text_type": "stego" if it["label"] else "normal",
                    "conditioned_on_input": True,
                    "perplexity": score["perplexity"],
                    "mean_nll": score["mean_nll"],
                    "num_tokens": score["num_tokens"],
                    "model": args.model,
                },
            )
            n_scored += 1
            if (i + 1) % 25 == 0 or i + 1 == len(pending):
                log.info(
                    "  [%s] %d/%d (last ppl=%.2f)",
                    system,
                    i + 1,
                    len(pending),
                    score["perplexity"],
                )

    log.info("Scored %d new texts", n_scored)
    summary_path = detection_metrics_dir(args.data_dir) / "perplexity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_summarize(read_jsonl(jsonl_path)), f, indent=2)
    log.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
