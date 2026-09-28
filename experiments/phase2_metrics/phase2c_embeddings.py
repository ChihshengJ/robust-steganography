"""Phase 2c — Signal 2: Embedding + MLP stegoanalysis.

Embeds each (task input x, text) of the detection set with OpenAI
``text-embedding-3-large`` (or, optionally, Qwen3-Embedding-8B served by
llama.cpp), then trains a 2-layer MLP on the leave-one-configuration-out
split by default (``stegoanalysis_common.loco_folds``), or ``--split matched``.
Early stopping uses a validation split of the training configurations, grouped
by input.

Embeddings are cached per (system, model) in an ``.npz`` keyed by uid and a
hash of the embedded string, so adding cells embeds only the new texts:

    data/experiments/phase2_metrics/detection/stegoanalysis/
        embeddings/emb_{system}_{model}.npz
        embedding_mlp_{model}_{system}_predictions.jsonl
        embedding_mlp_{system}.json

Usage:
    python -m experiments.phase2_metrics.phase2c_embeddings --systems story
    python -m experiments.phase2_metrics.phase2c_embeddings --models openai,qwen3
    python -m experiments.phase2_metrics.phase2c_embeddings --skip-classify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from experiments.utils.stegoanalysis_common import (
    DEFAULT_EMBEDDER_INSTRUCTION,
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
    with_input,
    write_jsonl,
)
from systems.env import load_env

load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "openai": {
        "backend": "openai",
        "model_name": "text-embedding-3-large",
        "slug": "text-embedding-3-large",
    },
    "qwen3": {
        "backend": "llamacpp",
        "model_name": "Qwen3-Embedding-8B-Q8_0",
        "slug": "Qwen3-Embedding-8B-Q8_0",
    },
}

# text-embedding-3-large takes at most 8192 tokens per input (cl100k_base).
OPENAI_MAX_TOKENS = 8191

MLP_CONFIG = {
    "hidden1": 256,
    "hidden2": 64,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
    "batch_size": 32,
}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _with_retries(fn, what: str, retries: int = 5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("  %s retry %d: %s", what, attempt + 1, e)
            time.sleep(2 ** (attempt + 1))


def embed_with_openai(
    strings: list[str], model_name: str, batch_size: int = 32
) -> np.ndarray:
    import openai
    import tiktoken

    client = openai.OpenAI()
    enc = tiktoken.get_encoding("cl100k_base")
    clipped = []
    for s in strings:
        ids = enc.encode(s)
        if len(ids) > OPENAI_MAX_TOKENS:
            log.warning("  input of %d tokens clipped to %d", len(ids), OPENAI_MAX_TOKENS)
            s = enc.decode(ids[:OPENAI_MAX_TOKENS])
        clipped.append(s)

    out: list[list[float]] = []
    for i in tqdm(range(0, len(clipped), batch_size), desc="openai embed", unit="batch"):
        chunk = clipped[i : i + batch_size]
        resp = _with_retries(
            lambda: client.embeddings.create(model=model_name, input=chunk),
            f"openai chunk {i}",
        )
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return np.asarray(out, dtype=np.float32)


def embed_with_llamacpp(
    strings: list[str],
    model_name: str,
    base_url: str,
    instruction: str,
    batch_size: int = 16,
) -> np.ndarray:
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="unused")
    formatted = [f"Instruct: {instruction}\nQuery: {s}" for s in strings]
    out: list[list[float]] = []
    for i in tqdm(range(0, len(formatted), batch_size), desc="llamacpp embed", unit="batch"):
        chunk = formatted[i : i + batch_size]
        resp = _with_retries(
            lambda: client.embeddings.create(model=model_name, input=chunk),
            f"llamacpp chunk {i}",
        )
        out.extend(item.embedding for item in resp.data)
    return np.asarray(out, dtype=np.float32)


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def embed_cached(items: list[dict], model_key: str, npz_path, args) -> np.ndarray:
    """Embeddings of every item's (input, text), in item order. Reuses cached
    rows whose uid and embedded string are unchanged; embeds the rest."""
    cfg = EMBEDDING_MODELS[model_key]
    strings = [with_input(it) for it in items]
    digests = [_digest(s) for s in strings]

    cached: dict[tuple[str, str], np.ndarray] = {}
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        for uid, dg, emb in zip(data["uids"], data["digests"], data["embeddings"]):
            cached[(str(uid), str(dg))] = emb

    todo = [i for i, it in enumerate(items) if (it["uid"], digests[i]) not in cached]
    log.info(
        "  %s: %d cached, %d to embed", cfg["slug"], len(items) - len(todo), len(todo)
    )
    if todo:
        new_strings = [strings[i] for i in todo]
        if cfg["backend"] == "openai":
            X_new = embed_with_openai(new_strings, cfg["model_name"])
        else:
            X_new = embed_with_llamacpp(
                new_strings,
                cfg["model_name"],
                args.llamacpp_base_url,
                args.embedder_instruction,
            )
        for i, emb in zip(todo, X_new):
            cached[(items[i]["uid"], digests[i])] = emb

    X = np.stack([cached[(it["uid"], dg)] for it, dg in zip(items, digests)])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cached)
    np.savez(
        npz_path,
        uids=np.array([k[0] for k in keys], dtype=object),
        digests=np.array([k[1] for k in keys], dtype=object),
        embeddings=np.stack([cached[k] for k in keys]),
        model_name=np.array(cfg["model_name"]),
    )
    return X


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
) -> EmbeddingMLP:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model = EmbeddingMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def mlp_predictions(items: list[dict], X: np.ndarray, split: str, detector: str) -> list[dict]:
    """Held-out P(stego) of an MLP per fold of the split."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    y = np.array([it["label"] for it in items])

    rows: list[dict] = []
    for fold in SPLITS[split](items):
        fit, val = inner_validation(items, fold["train"], seed=RANDOM_SEED + fold["fold"])
        scaler = StandardScaler().fit(X[fit])
        model = _train_mlp_fold(
            scaler.transform(X[fit]),
            y[fit],
            scaler.transform(X[val]),
            y[val],
            input_dim=X.shape[1],
            max_epochs=MLP_CONFIG["max_epochs"],
            patience=MLP_CONFIG["patience"],
            lr=MLP_CONFIG["lr"],
            weight_decay=MLP_CONFIG["weight_decay"],
            batch_size=MLP_CONFIG["batch_size"],
        )
        model.eval()
        with torch.no_grad():
            test_t = torch.tensor(
                scaler.transform(X[fold["test"]]), dtype=torch.float32
            ).to(device)
            proba = torch.softmax(model(test_t), dim=1)[:, 1].cpu().numpy()
        rows.extend(prediction_rows(items, fold, proba, detector))
        del model
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Signal 2: embedding + MLP classification"
    )
    add_common_args(parser)
    parser.add_argument(
        "--models",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["openai"],
        help=f"Comma-separated model keys: {', '.join(EMBEDDING_MODELS)} (default openai).",
    )
    parser.add_argument("--split", choices=sorted(SPLITS), default=DEFAULT_SPLIT)
    parser.add_argument(
        "--llamacpp-base-url",
        default="http://127.0.0.1:11435/v1",
        help="Base URL for the llama.cpp embeddings server (qwen3).",
    )
    parser.add_argument(
        "--embedder-instruction",
        default=DEFAULT_EMBEDDER_INSTRUCTION,
        help="Instruction prefix for Qwen3-Embedding-style models.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Only compute embeddings.",
    )
    args = parser.parse_args()

    for m in args.models:
        if m not in EMBEDDING_MODELS:
            parser.error(f"Unknown model key: {m}. Choices: {list(EMBEDDING_MODELS)}")

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)

    for system in args.systems:
        items = load_detection_set(args.data_dir, system, args.dirs)
        if not items:
            log.warning("[%s] no detection pairs found — skipping", system)
            continue
        log.info("=== embeddings / %s (%d texts) ===", system, len(items))

        models_result: dict = {}
        for model_key in args.models:
            cfg = EMBEDDING_MODELS[model_key]
            npz_path = out_dir / "embeddings" / f"emb_{system}_{cfg['slug']}.npz"
            X = embed_cached(items, model_key, npz_path, args)
            if args.skip_classify:
                continue
            detector = f"embedding_mlp_{cfg['slug']}"
            log.info("  %s: MLP (dim=%d, n=%d)", cfg["slug"], X.shape[1], X.shape[0])
            rows = mlp_predictions(items, X, args.split, detector)
            write_jsonl(predictions_path(args.data_dir, detector, system), rows)
            models_result[cfg["slug"]] = {
                "embedding_dim": int(X.shape[1]),
                "backend": cfg["backend"],
                "mlp_config": MLP_CONFIG,
                "metrics": summarize_predictions(rows),
            }

        if args.skip_classify:
            continue
        result = {
            "system": system,
            "input_view": "task input + text",
            "n_texts": len(items),
            "seed": RANDOM_SEED,
            "models": models_result,
        }
        out_path = out_dir / f"embedding_mlp_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
