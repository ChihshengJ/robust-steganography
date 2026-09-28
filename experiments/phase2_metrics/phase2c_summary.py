"""Phase 2c — Summary: aggregate all stegoanalysis signals.

Runs the perplexity-only detector inline (a logistic regression on the
input-conditioned GPT-2 perplexity from Phase 2b, through the same split as
the trained detectors), then reads every detector's out-of-fold predictions
(``{detector}_{system}_predictions.jsonl``) and writes one summary per system
with each detector's metrics per split (the leave-one-configuration-out axes
by default), pooled, per F, per test configuration and per (configuration, F):
accuracy and AUC with bootstrap CIs over inputs, and within-pair C2ST
p-values (``stegoanalysis_common.score_metrics``).

Usage:
    python -m experiments.phase2_metrics.phase2c_summary --systems story,litreview
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.phase2_metrics.phase2b_perplexity import perplexity_path
from experiments.utils.io import read_jsonl
from experiments.utils.stegoanalysis_common import (
    DEFAULT_SPLIT,
    RANDOM_SEED,
    SPLITS,
    add_common_args,
    load_detection_set,
    prediction_rows,
    predictions_path,
    seed_everything,
    stegoanalysis_dir,
    summarize_predictions,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PPL_DETECTOR = "perplexity_logreg"


def run_perplexity_only(
    items: list[dict], nll: dict[str, float], split: str
) -> list[dict]:
    """Held-out P(stego) of a logistic regression on mean NLL (log PPL)."""
    have = [it for it in items if it["uid"] in nll]
    if len(have) < len(items):
        log.warning(
            "  perplexity-only: %d of %d texts have no perplexity (run phase2b)",
            len(items) - len(have),
            len(items),
        )
    X = np.array([[nll[it["uid"]]] for it in have])
    y = np.array([it["label"] for it in have])
    rows: list[dict] = []
    for fold in SPLITS[split](have):
        scaler = StandardScaler().fit(X[fold["train"]])
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(scaler.transform(X[fold["train"]]), y[fold["train"]])
        proba = clf.predict_proba(scaler.transform(X[fold["test"]]))[:, 1]
        rows.extend(prediction_rows(have, fold, proba, PPL_DETECTOR))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Summary: aggregate all stegoanalysis signals"
    )
    add_common_args(parser)
    parser.add_argument("--split", choices=sorted(SPLITS), default=DEFAULT_SPLIT)
    parser.add_argument(
        "--ppl-model",
        default="gpt2-large",
        help="Phase 2b model whose perplexities the perplexity-only detector uses.",
    )
    args = parser.parse_args()

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)
    nll = {
        r["uid"]: r["mean_nll"]
        for r in read_jsonl(perplexity_path(args.data_dir))
        if r.get("model") == args.ppl_model and np.isfinite(r["mean_nll"])
    }

    for system in args.systems:
        items = load_detection_set(args.data_dir, system, args.dirs)
        if not items:
            log.warning("[%s] no detection pairs found — skipping", system)
            continue
        log.info("=== summary / %s (%d texts) ===", system, len(items))

        if nll:
            write_jsonl(
                predictions_path(args.data_dir, PPL_DETECTOR, system),
                run_perplexity_only(items, nll, args.split),
            )
        else:
            log.warning("  no %s perplexities found; run phase2b first", args.ppl_model)

        by_detector: dict[str, list[dict]] = defaultdict(list)
        for path in sorted(out_dir.glob(f"*_{system}_predictions.jsonl")):
            for r in read_jsonl(path):
                by_detector[r["detector"]].append(r)

        summary = {
            "system": system,
            "n_texts": len(items),
            "cells": sorted({it["cell"] for it in items}),
            "detectors": {
                d: summarize_predictions(rows) for d, rows in sorted(by_detector.items())
            },
        }
        for d, m in summary["detectors"].items():
            for split, block in m.items():
                p = block["pooled"]
                log.info(
                    "  %s [%s]: acc %.3f, AUC %s (n=%d)",
                    d,
                    split,
                    p["accuracy"],
                    f"{p['auc']:.3f}" if p["auc"] is not None else "n/a",
                    p["n"],
                )

        out_path = out_dir / f"summary_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
