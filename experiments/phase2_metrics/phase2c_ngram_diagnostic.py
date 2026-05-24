"""Phase 2c diagnostic — interpretable n-gram TF-IDF logistic stegoanalysis.

Answers "WHICH words/phrases separate stego (S) from a cover class?" so a
transformer's detection accuracy can be attributed to a concrete cause:
a surface prose / prompt-style artifact (launderable) vs content (citations).

5-fold stratified CV (TF-IDF fit per-fold, no leakage) gives the accuracy
estimate; a whole-data fit gives the discriminating n-gram coefficients and
each n-gram's document frequency within S and within the cover class.

Usage:
    python -m experiments.phase2_metrics.phase2c_ngram_diagnostic \
        --sub-experiment 2b --system litreview --truncate-words 374
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.phase2_metrics.phase2c_transformer import truncate_to_words
from experiments.utils.stegoanalysis_common import RANDOM_SEED, SUB_EXP_COVER, load_pair


def mask_years(text: str) -> str:
    """Replace every 4-digit publication year with a constant token, so a
    classifier can still see THAT a citation is present but not WHICH year.
    Isolates citation-recency from everything else."""
    return re.sub(r"\b(?:18|19|20)\d\d[a-z]?\b", "YEAR", text)


def _vectorizer(ngram_max: int, min_df: int) -> TfidfVectorizer:
    # Stop words are KEPT on purpose: meta-discourse phrases ("in this section",
    # "we contextualize") are exactly the prompt-artifact signal we're hunting.
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
    )


def cv_accuracy(texts: list[str], y: np.ndarray, ngram_max: int, min_df: int):
    """5-fold stratified CV; the TF-IDF vocabulary is fit per-fold on train only."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    X = np.array(texts, dtype=object)
    accs, f1s, aucs = [], [], []
    for tr, te in skf.split(X, y):
        vec = _vectorizer(ngram_max, min_df)
        X_tr, X_te = vec.fit_transform(X[tr]), vec.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        aucs.append(roc_auc_score(y[te], proba))
    return accs, f1s, aucs


def discriminating_ngrams(
    stego_texts: list[str],
    cover_texts: list[str],
    ngram_max: int,
    min_df: int,
    top_k: int,
):
    """Whole-data fit: return the top-k n-grams toward each class, with the
    fraction of stego docs and cover docs that contain each n-gram."""
    texts = stego_texts + cover_texts
    y = np.array([1] * len(stego_texts) + [0] * len(cover_texts))  # 1=stego, 0=cover
    vec = _vectorizer(ngram_max, min_df)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]

    # per-class document frequency for each feature, same tokenization
    df_s = np.asarray((vec.transform(stego_texts) > 0).mean(axis=0)).ravel()
    df_c = np.asarray((vec.transform(cover_texts) > 0).mean(axis=0)).ravel()

    order = np.argsort(coef)

    def pack(idxs):
        return [(names[i], coef[i], df_s[i], df_c[i]) for i in idxs]

    toward_cover = pack(order[:top_k])        # most negative -> class 0 (cover)
    toward_stego = pack(order[::-1][:top_k])  # most positive -> class 1 (stego)
    return toward_stego, toward_cover, len(names)


def _print_table(title: str, rows) -> None:
    print(f"\n{title}")
    print(f"  {'n-gram':<36} {'coef':>7}  {'S df':>6}  {'cover df':>9}")
    print(f"  {'-' * 36} {'-' * 7}  {'-' * 6}  {'-' * 9}")
    for name, coef, df_s, df_c in rows:
        print(f"  {name:<36} {coef:>7.2f}  {df_s * 100:>5.0f}%  {df_c * 100:>8.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="n-gram TF-IDF stegoanalysis diagnostic"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--sub-experiment", choices=["2a", "2b", "2c"], default="2b"
    )
    parser.add_argument("--system", default="litreview")
    parser.add_argument(
        "--truncate-words",
        type=int,
        default=None,
        help="Truncate every text to its first N words before vectorizing "
        "(use ~374 to mirror the transformer's 512-token window for litreview).",
    )
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--mask-years",
        action="store_true",
        help="Replace 4-digit years with a constant token before vectorizing, "
        "to test how much of the signal is citation recency.",
    )
    args = parser.parse_args()

    cover_type = SUB_EXP_COVER[args.sub_experiment]
    stego_recs, cover_recs = load_pair(args.data_dir, args.system, args.sub_experiment)
    stego_texts = [r["text"] for r in stego_recs]
    cover_texts = [r["text"] for r in cover_recs]
    if args.mask_years:
        stego_texts = [mask_years(t) for t in stego_texts]
        cover_texts = [mask_years(t) for t in cover_texts]
    if args.truncate_words:
        stego_texts = [truncate_to_words(t, args.truncate_words) for t in stego_texts]
        cover_texts = [truncate_to_words(t, args.truncate_words) for t in cover_texts]

    texts = stego_texts + cover_texts
    y = np.array([1] * len(stego_texts) + [0] * len(cover_texts))

    print("=" * 74)
    print(
        f"n-gram diagnostic: {args.sub_experiment} / {args.system}  "
        f"(S vs {cover_type})"
    )
    print(
        f"  n_stego={len(stego_texts)}  n_cover={len(cover_texts)}  "
        f"ngram_range=(1,{args.ngram_max})  min_df={args.min_df}  "
        f"truncate_words={args.truncate_words}"
    )
    print("=" * 74)

    accs, f1s, aucs = cv_accuracy(texts, y, args.ngram_max, args.min_df)
    print(
        f"\n5-fold CV  accuracy={np.mean(accs):.3f}+-{np.std(accs):.3f}  "
        f"macroF1={np.mean(f1s):.3f}  AUC={np.mean(aucs):.3f}"
    )

    toward_stego, toward_cover, vocab = discriminating_ngrams(
        stego_texts, cover_texts, args.ngram_max, args.min_df, args.top_k
    )
    print(f"vocab size: {vocab}")
    print("\n('S df' / 'cover df' = % of stego / cover docs containing the n-gram)")
    _print_table(
        f"TOP {args.top_k} n-grams pushing toward COVER ({cover_type}):", toward_cover
    )
    _print_table(f"TOP {args.top_k} n-grams pushing toward STEGO:", toward_stego)


if __name__ == "__main__":
    main()
