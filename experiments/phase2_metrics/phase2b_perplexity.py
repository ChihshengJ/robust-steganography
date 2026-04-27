"""Phase 2b: Perplexity (gpt2-large by default).

Reads all Phase 1 JSONL outputs at ``data/experiments/phase1_texts/`` and
emits:

    data/experiments/phase2_metrics/perplexity_all.jsonl
    data/experiments/phase2_metrics/perplexity_summary.json

Perplexity is computed via ``experiments.utils.perplexity.PerplexityScorer``
(sliding-window NLL over a HuggingFace causal LM, mean over tokens, exp).
The scorer auto-selects ``cuda`` -> ``mps`` -> ``cpu``.

The output JSONL is checkpoint-resumable: ids already present are skipped.
This matters because gpt2-large over ~2700 texts on CPU/MPS is slow.

Usage:
    python -m experiments.phase2_metrics.phase2b_perplexity
    python -m experiments.phase2_metrics.phase2b_perplexity --systems topicqa --limit 10
    python -m experiments.phase2_metrics.phase2b_perplexity --model gpt2  # smaller, faster
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from experiments.utils.io import append_jsonl, load_completed_ids, read_jsonl
from experiments.utils.perplexity import PerplexityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEMS = ["topicqa", "story", "litreview"]
TEXT_TYPES = ["stego", "cover_c1", "cover_c2"]


def _phase1_path(data_dir: Path, system: str, text_type: str) -> Path:
    return data_dir / "phase1_texts" / f"{system}_{text_type}.jsonl"


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


def _summarize(jsonl_path: Path) -> dict:
    summary: dict[str, dict] = {}
    by_group: dict[tuple[str, str], list[float]] = {}
    for rec in read_jsonl(jsonl_path):
        ppl = rec.get("perplexity")
        if ppl is None or ppl == float("inf"):
            continue
        by_group.setdefault((rec["system"], rec["text_type"]), []).append(ppl)
    for (system, text_type), values in by_group.items():
        summary.setdefault(system, {})[text_type] = _stat_block(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2b: perplexity scoring")
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--systems",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=SYSTEMS,
    )
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap records per (system, text_type) — for smoke tests.",
    )
    args = parser.parse_args()

    out_dir = args.data_dir / "phase2_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "perplexity_all.jsonl"
    summary_path = out_dir / "perplexity_summary.json"

    completed_ids = load_completed_ids(jsonl_path)
    log.info("Found %d already-scored records in %s", len(completed_ids), jsonl_path)

    log.info("Loading PerplexityScorer (model=%s, device=%s)", args.model, args.device or "auto")
    scorer = PerplexityScorer(model_name=args.model, device=args.device)
    log.info("Scorer ready on device=%s", scorer.device)

    n_scored = 0
    n_skipped = 0
    for system in args.systems:
        for text_type in TEXT_TYPES:
            src = _phase1_path(args.data_dir, system, text_type)
            phase1_records = read_jsonl(src)
            if not phase1_records:
                log.warning("No records found at %s", src)
                continue

            if args.limit is not None:
                phase1_records = phase1_records[: args.limit]

            log.info("[%s/%s] %d candidate records", system, text_type, len(phase1_records))
            for i, rec in enumerate(phase1_records):
                rid = rec["id"]
                if rid in completed_ids:
                    n_skipped += 1
                    continue

                text = rec["text"]
                if not text or not text.strip():
                    log.warning("Empty text for %s — skipping", rid)
                    continue

                score = scorer.score(text)
                out = {
                    "id": rid,
                    "system": rec["system"],
                    "text_type": rec["text_type"],
                    "prompt_idx": rec.get("prompt_idx"),
                    "perplexity": score["perplexity"],
                    "mean_nll": score["mean_nll"],
                    "num_tokens": score["num_tokens"],
                    "model": args.model,
                }
                append_jsonl(jsonl_path, out)
                completed_ids.add(rid)
                n_scored += 1
                if (i + 1) % 25 == 0 or (i + 1) == len(phase1_records):
                    log.info(
                        "  [%s/%s] %d/%d (last ppl=%.2f)",
                        system, text_type, i + 1, len(phase1_records), score["perplexity"],
                    )

    log.info("Scored %d new records (%d already done)", n_scored, n_skipped)

    summary = _summarize(jsonl_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
