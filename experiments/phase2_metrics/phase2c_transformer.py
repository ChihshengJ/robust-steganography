"""Phase 2c — Signal 1: Transformer fine-tuning for stegoanalysis.

5-fold stratified CV with DistilBERT (or any HuggingFace sequence classifier).
Reports accuracy, macro-F1, and AUC per fold plus aggregated mean/std.

Usage:
    python -m experiments.phase2_metrics.phase2c_transformer \
        --sub-experiment 2a --systems topicqa \
        --transformer distilbert-base-uncased --max-epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.utils.stegoanalysis_common import (
    RANDOM_SEED,
    SUB_EXP_COVER,
    add_common_args,
    agg_folds,
    iter_tasks,
    load_pair,
    seed_everything,
    stegoanalysis_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_truncate_words(value: str):
    """argparse type for --truncate-words: an int word budget, or 'auto'."""
    if value.lower() == "auto":
        return "auto"
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--truncate-words must be positive or 'auto'")
    return n


def truncate_to_words(text: str, n: int) -> str:
    """Return `text` cut to its first `n` whitespace-delimited words.

    Preserves the original whitespace/newlines up to the end of the n-th word
    (so paragraph breaks survive); texts with <= n words are returned unchanged.
    """
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n:
        return text
    return text[: matches[n - 1].end()]


def run_transformer(
    texts: list[str],
    y: np.ndarray,
    model_name: str,
    max_epochs: int,
    batch_size: int,
    max_length: int,
) -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class _Ds(Dataset):
        def __init__(self, txts, labels):
            self.enc = tokenizer(
                txts, truncation=True, padding="max_length", max_length=max_length
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                "labels": torch.tensor(int(self.labels[i])),
            }

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="macro")),
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    texts_arr = np.array(texts, dtype=object)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts_arr, y)):
        log.info(
            "  transformer fold %d/5 (n_train=%d n_test=%d)",
            fold_idx + 1,
            len(train_idx),
            len(test_idx),
        )
        train_ds = _Ds(texts_arr[train_idx].tolist(), y[train_idx])
        test_ds = _Ds(texts_arr[test_idx].tolist(), y[test_idx])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

        out_dir = Path(".cache/phase2c_transformer") / f"fold_{fold_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=max_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=20,
            save_total_limit=1,
            report_to=[],
            seed=RANDOM_SEED,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        train_out = trainer.train()
        eval_out = trainer.evaluate()

        pred_logits = trainer.predict(test_ds).predictions
        proba = torch.softmax(torch.tensor(pred_logits), dim=1).numpy()[:, 1]
        preds = np.argmax(pred_logits, axis=1)
        folds.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": float(accuracy_score(y[test_idx], preds)),
                "f1": float(f1_score(y[test_idx], preds, average="macro")),
                "auc": (
                    float(roc_auc_score(y[test_idx], proba))
                    if len(np.unique(y[test_idx])) > 1
                    else None
                ),
                "train_loss": float(train_out.training_loss)
                if train_out.training_loss
                else None,
                "eval_loss": float(eval_out.get("eval_loss", float("nan"))),
            }
        )

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = agg_folds(folds)
    result["model"] = model_name
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Signal 1: transformer fine-tuning"
    )
    add_common_args(parser)
    parser.add_argument("--transformer", default="distilbert-base-uncased")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncate-words",
        type=_parse_truncate_words,
        default=None,
        metavar="N|auto",
        help=(
            "Length control: truncate every text in BOTH classes to the first "
            "N words before classification, so the model cannot exploit a "
            "length gap between S and the cover. 'auto' uses the minimum word "
            "count across the loaded pair (every text becomes exactly that "
            "length). Default: no truncation."
        ),
    )
    args = parser.parse_args()

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)

    for sub_exp, system in iter_tasks(args):
        cover_type = SUB_EXP_COVER[sub_exp]
        log.info("=== transformer %s / %s (S vs %s) ===", sub_exp, system, cover_type)

        stego_recs, cover_recs = load_pair(args.data_dir, system, sub_exp)
        if not stego_recs or not cover_recs:
            log.warning("Missing data for %s/%s — skipping", sub_exp, system)
            continue

        stego_texts = [r["text"] for r in stego_recs]
        cover_texts = [r["text"] for r in cover_recs]

        # Optional length control: truncate every text in both classes to a
        # common word budget so the classifier cannot exploit a length gap
        # between S and the cover (e.g. LitReview C2 overshoots its word
        # target by ~24%, which alone separates the classes at ~0.78 acc).
        trunc_info = None
        if args.truncate_words is not None:
            all_texts = stego_texts + cover_texts
            if args.truncate_words == "auto":
                n_words = min(len(t.split()) for t in all_texts)
            else:
                n_words = args.truncate_words
            wc_before = float(np.mean([len(t.split()) for t in all_texts]))
            stego_texts = [truncate_to_words(t, n_words) for t in stego_texts]
            cover_texts = [truncate_to_words(t, n_words) for t in cover_texts]
            wc_after = float(
                np.mean([len(t.split()) for t in stego_texts + cover_texts])
            )
            trunc_info = {
                "requested": args.truncate_words,
                "n_words": n_words,
                "mean_words_before": wc_before,
                "mean_words_after": wc_after,
            }
            log.info(
                "Length control: truncated all texts to %d words "
                "(mean words %.0f -> %.0f)",
                n_words,
                wc_before,
                wc_after,
            )

        texts = stego_texts + cover_texts
        y = np.array([1] * len(stego_recs) + [0] * len(cover_recs))

        result = run_transformer(
            texts,
            y,
            model_name=args.transformer,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        result.update(
            {
                "sub_experiment": sub_exp,
                "system": system,
                "cover_type": cover_type,
                "n_stego": len(stego_recs),
                "n_cover": len(cover_recs),
                "truncate_words": trunc_info,
                "seed": RANDOM_SEED,
            }
        )

        suffix = f"_trunc{trunc_info['n_words']}" if trunc_info else ""
        out_path = out_dir / f"transformer_{sub_exp}_{system}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
