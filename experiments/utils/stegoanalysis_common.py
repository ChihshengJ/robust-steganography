"""Shared constants and helpers for the phase 2c stegoanalysis scripts."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.utils.io import read_jsonl

SYSTEMS = ["topicqa", "story", "litreview"]
SUB_EXP_COVER = {"2a": "cover_c1", "2b": "cover_c2", "2c": "cover_c3"}
SUB_EXP_SYSTEMS = {
    "2a": ["topicqa", "story", "litreview"],
    "2b": ["topicqa", "story", "litreview"],
    # 2c (Option B: Qwen outline + GPT-4.1 synthesis cover) — story only.
    "2c": ["story"],
}
RANDOM_SEED = 42

DEFAULT_EMBEDDER_INSTRUCTION = (
    "Given a passage of AI-generated text, encode it to capture stylistic, "
    "structural, and content-selection patterns useful for distinguishing "
    "steganographically-encoded text from ordinary generated text."
)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def phase1_path(data_dir: Path, system: str, text_type: str) -> Path:
    return data_dir / "phase1_texts" / f"{system}_{text_type}.jsonl"


def load_pair(
    data_dir: Path, system: str, sub_exp: str
) -> tuple[list[dict], list[dict]]:
    """Return (stego_records, cover_records) sorted by prompt_idx and aligned."""
    cover_type = SUB_EXP_COVER[sub_exp]
    stego = read_jsonl(phase1_path(data_dir, system, "stego"))
    cover = read_jsonl(phase1_path(data_dir, system, cover_type))

    stego.sort(key=lambda r: r.get("prompt_idx", 0))
    cover.sort(key=lambda r: r.get("prompt_idx", 0))

    stego_by_idx = {r["prompt_idx"]: r for r in stego if "prompt_idx" in r}
    cover_by_idx = {r["prompt_idx"]: r for r in cover if "prompt_idx" in r}
    common = sorted(set(stego_by_idx) & set(cover_by_idx))

    aligned_stego = [stego_by_idx[i] for i in common]
    aligned_cover = [cover_by_idx[i] for i in common]
    return aligned_stego, aligned_cover


def stat(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def agg_folds(folds: list[dict]) -> dict:
    keys = ["accuracy", "f1", "auc"]
    mean = {
        k: stat([f[k] for f in folds if f.get(k) is not None])["mean"] for k in keys
    }
    std = {k: stat([f[k] for f in folds if f.get(k) is not None])["std"] for k in keys}
    return {"folds": folds, "mean": mean, "std": std}


def cv_logreg(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        folds.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "f1": float(f1_score(y[test_idx], pred, average="macro")),
                "auc": (
                    float(roc_auc_score(y[test_idx], proba))
                    if len(np.unique(y[test_idx])) > 1
                    else None
                ),
            }
        )
    return agg_folds(folds)


def stegoanalysis_dir(data_dir: Path) -> Path:
    d = data_dir / "phase2_metrics" / "stegoanalysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--sub-experiment",
        choices=["2a", "2b", "2c", "both", "all"],
        default="both",
        help="'both' = 2a+2b (back-compat); 'all' = 2a+2b+2c.",
    )
    parser.add_argument(
        "--systems",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=SYSTEMS,
    )


def iter_tasks(args) -> Iterator[tuple[str, str]]:
    """Yield valid (sub_exp, system) pairs from parsed args."""
    if args.sub_experiment == "both":
        sub_exps = ["2a", "2b"]
    elif args.sub_experiment == "all":
        sub_exps = ["2a", "2b", "2c"]
    else:
        sub_exps = [args.sub_experiment]
    for sub_exp in sub_exps:
        for system in args.systems:
            if system in SUB_EXP_SYSTEMS[sub_exp]:
                yield sub_exp, system
