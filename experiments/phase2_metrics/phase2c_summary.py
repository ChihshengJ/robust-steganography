"""Phase 2c — Summary: aggregate all stegoanalysis signals.

Reads results from the transformer, embedding MLP, and LLM judge scripts,
runs the perplexity-only logistic regression inline (trivial single-feature
classifier using GPT-2 PPL from Phase 2b), and writes a unified summary JSON.

Usage:
    python -m experiments.phase2_metrics.phase2c_summary \
        --sub-experiment both --systems topicqa,story,litreview
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experiments.utils.io import read_jsonl
from experiments.utils.stegoanalysis_common import (
    RANDOM_SEED,
    SUB_EXP_COVER,
    add_common_args,
    cv_logreg,
    iter_tasks,
    load_pair,
    seed_everything,
    stegoanalysis_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_perplexity_map(data_dir: Path) -> dict[str, float]:
    path = data_dir / "phase2_metrics" / "perplexity_all.jsonl"
    records = read_jsonl(path)
    if not records:
        return {}
    return {r["id"]: r["perplexity"] for r in records if "perplexity" in r}


def run_perplexity_only(
    stego_recs: list[dict],
    cover_recs: list[dict],
    ppl_map: dict[str, float],
) -> dict:
    pairs = []
    missing = 0
    for label, recs in [(1, stego_recs), (0, cover_recs)]:
        for r in recs:
            ppl = ppl_map.get(r["id"])
            if ppl is None or ppl == float("inf"):
                missing += 1
                continue
            pairs.append((ppl, label))
    if missing:
        log.warning("  perplexity-only: %d records missing perplexity values", missing)

    if not pairs:
        return {"error": "no perplexity values available"}

    X = np.array([[p[0]] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return cv_logreg(X, y)


def build_summary(
    data_dir: Path,
    out_dir: Path,
    sub_exp: str,
    system: str,
    ppl_map: dict[str, float],
) -> dict:
    cover_type = SUB_EXP_COVER[sub_exp]
    stego_recs, cover_recs = load_pair(data_dir, system, sub_exp)

    result: dict = {
        "sub_experiment": sub_exp,
        "system": system,
        "cover_type": cover_type,
        "n_stego": len(stego_recs),
        "n_cover": len(cover_recs),
        "seed": RANDOM_SEED,
    }

    # Transformer
    transformer = _load_json(out_dir / f"transformer_{sub_exp}_{system}.json")
    result["transformer"] = transformer if transformer else {"status": "not_available"}

    # Embedding MLP
    emb_mlp = _load_json(out_dir / f"embedding_mlp_{sub_exp}_{system}.json")
    if emb_mlp and "models" in emb_mlp:
        result["embedding_mlp"] = emb_mlp["models"]
    else:
        result["embedding_mlp"] = {"status": "not_available"}

    # Perplexity-only logreg
    if ppl_map and stego_recs and cover_recs:
        result["perplexity_only"] = run_perplexity_only(stego_recs, cover_recs, ppl_map)
    else:
        result["perplexity_only"] = {"status": "not_available"}

    # LLM judge
    judge = _load_json(
        out_dir / f"llm_judge_{sub_exp}_{system}_anthropic_claude-sonnet-4.5.json"
    )
    result["llm_judge"] = judge if judge else {"status": "not_available"}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Summary: aggregate all stegoanalysis signals"
    )
    add_common_args(parser)
    args = parser.parse_args()

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)

    ppl_map = _load_perplexity_map(args.data_dir)
    if ppl_map:
        log.info("Loaded %d perplexity records", len(ppl_map))
    else:
        log.warning("No perplexity data found; perplexity-only signal will be skipped")

    for sub_exp, system in iter_tasks(args):
        log.info("=== summary %s / %s ===", sub_exp, system)
        result = build_summary(args.data_dir, out_dir, sub_exp, system, ppl_map)

        out_path = out_dir / f"summary_{sub_exp}_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
