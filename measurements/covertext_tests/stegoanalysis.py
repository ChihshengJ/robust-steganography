"""
Input: JSON file with {"stego": [...], "cover": [...]}
Output: JSON with classifier metrics (accuracy, F1, AUC) and perplexity comparison.

Usage:
    python exp2_stegoanalysis.py --input data.json --output results.json
    python exp2_stegoanalysis.py --input data.json --classifier-model roberta-base --ppl-model gpt2-large
"""

import argparse
import json

import numpy as np
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .utils import compute_perplexities


def summarize(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def run_classifier(
    stego_texts, cover_texts, embedding_model="all-mpnet-base-v2", n_folds=5
):
    print(f"Encoding texts with {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    all_texts = stego_texts + cover_texts
    labels = np.array([1] * len(stego_texts) + [0] * len(cover_texts))
    embeddings = model.encode(all_texts, show_progress_bar=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_results.append(
            {
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "auc": float(roc_auc_score(y_test, y_prob)),
            }
        )
        print(
            f"  Fold {fold_idx}: acc={fold_results[-1]['accuracy']:.3f} f1={fold_results[-1]['f1']:.3f} auc={fold_results[-1]['auc']:.3f}"
        )

    avg = {
        "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "f1": float(np.mean([r["f1"] for r in fold_results])),
        "auc": float(np.mean([r["auc"] for r in fold_results])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_results])),
        "f1_std": float(np.std([r["f1"] for r in fold_results])),
        "auc_std": float(np.std([r["auc"] for r in fold_results])),
    }

    return {"folds": fold_results, "average": avg}


def run_perplexity_comparison(stego_texts, cover_texts, model_name="gpt2-large"):
    print(f"Computing perplexity for stego texts ({len(stego_texts)})...")
    stego_ppl = compute_perplexities(stego_texts, model_name=model_name)
    print(f"Computing perplexity for cover texts ({len(cover_texts)})...")
    cover_ppl = compute_perplexities(cover_texts, model_name=model_name)

    stego_clean = [p for p in stego_ppl if not np.isnan(p)]
    cover_clean = [p for p in cover_ppl if not np.isnan(p)]

    stat, pvalue = (
        mannwhitneyu(stego_clean, cover_clean, alternative="two-sided")
        if stego_clean and cover_clean
        else (None, None)
    )

    return {
        "stego": {"values": stego_ppl, **summarize(stego_ppl)},
        "cover": {"values": cover_ppl, **summarize(cover_ppl)},
        "mann_whitney_u": float(stat) if stat is not None else None,
        "mann_whitney_p": float(pvalue) if pvalue is not None else None,
    }


def run(stego_texts, cover_texts, ppl_model="gpt2-large"):
    classifier_results = run_classifier(stego_texts, cover_texts)
    perplexity_results = run_perplexity_comparison(
        stego_texts, cover_texts, model_name=ppl_model
    )
    return {"classifier": classifier_results, "perplexity": perplexity_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="exp2_results.json")
    parser.add_argument("--ppl-model", default="gpt2-large")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    stego = data["stego"]
    cover = data["cover"]

    results = run(stego, cover, ppl_model=args.ppl_model)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
