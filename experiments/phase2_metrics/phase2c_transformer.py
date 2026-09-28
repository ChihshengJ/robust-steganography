"""Phase 2c — Signal 1: Transformer fine-tuning for stegoanalysis.

Fine-tunes DistilBERT (or any HuggingFace sequence classifier) on the
detection set: each example is the pair (task input x, text), encoded as a
sentence pair so x is kept whole and only the text is truncated. Runs the
leave-one-configuration-out split by default (``stegoanalysis_common.loco_folds``:
held-out writer, G, or both), or ``--split matched``; early stopping uses a
validation split of the training configurations, grouped by input, never the
test configuration.

Writes held-out predictions and their metrics:

    data/experiments/phase2_metrics/detection/stegoanalysis/
        transformer_{system}_predictions.jsonl
        transformer_{system}.json

Usage:
    python -m experiments.phase2_metrics.phase2c_transformer \
        --systems story --transformer distilbert-base-uncased --max-epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

from experiments.utils.stegoanalysis_common import (
    DEFAULT_SPLIT,
    RANDOM_SEED,
    SPLITS,
    add_common_args,
    inner_validation,
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

DETECTOR = "transformer"


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


def train_and_predict(
    items: list[dict],
    fold: dict,
    texts: list[str],
    model_name: str,
    max_epochs: int,
    batch_size: int,
    max_length: int,
    cache_dir: Path,
) -> np.ndarray:
    """Fine-tune on a fold's training texts; P(stego) for its test texts."""
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
        def __init__(self, idx):
            self.enc = tokenizer(
                [items[i]["input"] for i in idx],
                [texts[i] for i in idx],
                truncation="only_second",
                padding="max_length",
                max_length=max_length,
            )
            self.labels = [items[i]["label"] for i in idx]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                "labels": torch.tensor(int(self.labels[i])),
            }

    fit, val = inner_validation(items, fold["train"], seed=RANDOM_SEED + fold["fold"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=str(cache_dir),
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
        train_dataset=_Ds(fit),
        eval_dataset=_Ds(val),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    logits = trainer.predict(_Ds(fold["test"])).predictions
    proba = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Signal 1: transformer fine-tuning"
    )
    add_common_args(parser)
    parser.add_argument("--transformer", default="distilbert-base-uncased")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--split", choices=sorted(SPLITS), default=DEFAULT_SPLIT)
    parser.add_argument(
        "--truncate-words",
        type=_parse_truncate_words,
        default=None,
        metavar="N|auto",
        help=(
            "Length control: truncate every text (not the input) to its first N "
            "words before classification. 'auto' uses the shortest text of the "
            "system. Default: no truncation (the classes are length matched)."
        ),
    )
    args = parser.parse_args()

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)

    for system in args.systems:
        items = load_detection_set(args.data_dir, system, args.dirs)
        if not items:
            log.warning("[%s] no detection pairs found — skipping", system)
            continue
        texts = [it["text"] for it in items]
        trunc_info = None
        if args.truncate_words is not None:
            n_words = (
                min(len(t.split()) for t in texts)
                if args.truncate_words == "auto"
                else args.truncate_words
            )
            texts = [truncate_to_words(t, n_words) for t in texts]
            trunc_info = {"requested": args.truncate_words, "n_words": n_words}
            log.info("[%s] truncated every text to %d words", system, n_words)
        detector = DETECTOR + (f"_trunc{trunc_info['n_words']}" if trunc_info else "")

        rows: list[dict] = []
        for fold in SPLITS[args.split](items):
            log.info(
                "[%s] %s train on %s, fold %d (train %d, test %d)",
                system,
                fold["split"],
                fold["train_config"],
                fold["fold"],
                len(fold["train"]),
                len(fold["test"]),
            )
            cache_dir = (
                Path(".cache/phase2c_transformer")
                / system
                / fold["split"]
                / f"fold_{fold['fold']}"
            )
            proba = train_and_predict(
                items,
                fold,
                texts,
                args.transformer,
                args.max_epochs,
                args.batch_size,
                args.max_length,
                cache_dir,
            )
            rows.extend(prediction_rows(items, fold, proba, detector))

        write_jsonl(predictions_path(args.data_dir, detector, system), rows)
        result = {
            "detector": detector,
            "system": system,
            "model": args.transformer,
            "input_view": "task input + text (sentence pair)",
            "n_texts": len(items),
            "truncate_words": trunc_info,
            "seed": RANDOM_SEED,
            "metrics": summarize_predictions(rows),
        }
        out_path = out_dir / f"{detector}_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
