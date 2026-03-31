"""
Experiment 5: Text Quality

Input: JSON file with {"texts": [...]} or {"texts": [...], "labels": [...]}
       If labels provided, computes per-group stats.
Output: JSON with per-text perplexity and aggregate stats.

Usage:
    python exp5_text_quality.py --input texts.json --output results.json
    python exp5_text_quality.py --input texts.json --model gpt2-large
"""

import argparse
import json

import numpy as np

from .utils import compute_perplexities


def summarize(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run(texts, labels=None, model_name="gpt2-large"):
    print(f"Computing perplexity with {model_name} for {len(texts)} texts...")
    perplexities = compute_perplexities(texts, model_name=model_name)

    per_text = [{"index": i, "perplexity": p} for i, p in enumerate(perplexities)]

    result = {
        "model": model_name,
        "per_text": per_text,
        "aggregate": summarize(perplexities),
    }

    if labels is not None:
        groups = {}
        for ppl, label in zip(perplexities, labels):
            groups.setdefault(label, []).append(ppl)
        result["by_group"] = {label: summarize(vals) for label, vals in groups.items()}

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="exp5_results.json")
    parser.add_argument("--model", default="gpt2-large")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    texts = data["texts"]
    labels = data.get("labels")
    if labels is not None:
        assert len(texts) == len(labels), "Mismatched lengths"

    results = run(texts, labels=labels, model_name=args.model)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
