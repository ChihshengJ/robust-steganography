"""
Experiment 1: Attack Validation

Input: JSON file with {"originals": [...], "attacked": [...]}
Output: JSON with per-pair and aggregate metrics (BERTScore F1, cosine similarity, BLEU, TER)

Usage:
    python exp1_attack_validation.py --input data.json --output results.json
"""

import argparse
import json

import numpy as np
import sacrebleu
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def compute_bertscore(originals, attacked):
    P, R, F1 = bert_score(attacked, originals, lang="en", verbose=True)
    return F1.tolist()


def compute_cosine_similarity(originals, attacked, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    emb_orig = model.encode(originals)
    emb_attack = model.encode(attacked)
    sims = [float(cos_sim([a], [b])[0, 0]) for a, b in zip(emb_orig, emb_attack)]
    return sims


def compute_bleu(originals, attacked):
    scores = []
    for orig, att in zip(originals, attacked):
        result = sacrebleu.sentence_bleu(att, [orig])
        scores.append(result.score / 100.0)
    return scores


def summarize(values, name):
    arr = np.array(values)
    return {
        f"{name}_mean": float(np.nanmean(arr)),
        f"{name}_std": float(np.nanstd(arr)),
        f"{name}_min": float(np.nanmin(arr)),
        f"{name}_max": float(np.nanmax(arr)),
    }


def run(originals, attacked):
    print("Computing BERTScore...")
    bertscore_f1 = compute_bertscore(originals, attacked)

    print("Computing cosine similarity...")
    cosine_sims = compute_cosine_similarity(originals, attacked)

    print("Computing BLEU...")
    bleu_scores = compute_bleu(originals, attacked)

    per_pair = []
    for i in range(len(originals)):
        per_pair.append(
            {
                "index": i,
                "bertscore_f1": bertscore_f1[i],
                "cosine_similarity": cosine_sims[i],
                "bleu": bleu_scores[i],
            }
        )

    aggregate = {
        **summarize(bertscore_f1, "bertscore_f1"),
        **summarize(cosine_sims, "cosine_similarity"),
        **summarize(bleu_scores, "bleu"),
    }

    return {"per_pair": per_pair, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="exp1_results.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    originals = data["originals"]
    attacked = data["attacked"]
    assert len(originals) == len(attacked), "Mismatched lengths"

    results = run(originals, attacked)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
