"""Phase 2c — Signal 2: Embedding generation + MLP classification.

Generates embeddings from two models (Qwen3-Embedding-8B via llama.cpp and
text-embedding-005 via Google REST API), saves them as ``.npz`` files, then
trains 2-layer MLP classifiers with 5-fold stratified CV.

Usage:
    # Full run (both models):
    python -m experiments.phase2_metrics.phase2c_embeddings \
        --sub-experiment 2a --systems topicqa --models qwen3,google

    # Only Qwen3 embeddings + classification:
    python -m experiments.phase2_metrics.phase2c_embeddings \
        --sub-experiment 2a --systems topicqa --models qwen3

    # Re-run MLP on existing embeddings (skip embedding step):
    python -m experiments.phase2_metrics.phase2c_embeddings \
        --sub-experiment 2a --systems topicqa --skip-embed

    # Only generate embeddings (skip classification):
    python -m experiments.phase2_metrics.phase2c_embeddings \
        --sub-experiment 2a --systems topicqa --skip-classify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from experiments.utils.stegoanalysis_common import (
    DEFAULT_EMBEDDER_INSTRUCTION,
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

EMBEDDING_MODELS = {
    "qwen3": {
        "backend": "llamacpp",
        "model_name": "Qwen3-Embedding-8B-Q8_0",
        "slug": "Qwen3-Embedding-8B-Q8_0",
    },
    "google": {
        "backend": "google",
        "model_name": "text-embedding-005",
        "slug": "text-embedding-005",
    },
}


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def embed_with_llamacpp(
    texts: list[str],
    model_name: str,
    base_url: str,
    instruction: str,
    batch_size: int = 16,
) -> np.ndarray:
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="unused")
    formatted = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

    embeddings: list[list[float]] = []
    with tqdm(total=len(formatted), desc="llamacpp embed", unit="txt") as pbar:
        for i in range(0, len(formatted), batch_size):
            chunk = formatted[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = client.embeddings.create(model=model_name, input=chunk)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    log.warning("  llamacpp retry %d (chunk %d): %s", attempt + 1, i, e)
                    time.sleep(2**attempt)
            embeddings.extend(item.embedding for item in resp.data)
            pbar.update(len(chunk))

    return np.asarray(embeddings, dtype=np.float32)


def _texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _load_google_cache(
    cache_path: Path, fingerprint: str, n: int
) -> dict[int, list[float]]:
    """Load a JSONL resume cache. First line is a header; remaining lines are
    {"idx": int, "emb": [...]}. If the header doesn't match, the cache is
    discarded (returned empty)."""
    completed: dict[int, list[float]] = {}
    if not cache_path.exists():
        return completed

    with open(cache_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return completed
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError:
            log.warning("  cache header unreadable, ignoring %s", cache_path)
            return completed
        if header.get("fingerprint") != fingerprint or header.get("n") != n:
            log.warning(
                "  cache fingerprint mismatch at %s — starting fresh", cache_path
            )
            return completed
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed[int(rec["idx"])] = rec["emb"]
            except (json.JSONDecodeError, KeyError, ValueError):
                # Last line may be partially written if we crashed mid-flush.
                log.warning("  skipping malformed cache line")
                continue
    return completed


def embed_with_google(
    texts: list[str],
    model_name: str = "text-embedding-005",
    task_type: str = "CLASSIFICATION",
    sleep_between: float = 0.1,
    max_retries: int = 5,
    cache_path: Path | None = None,
) -> np.ndarray:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required for text-embedding-005"
        )

    url = (
        f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:predict?key={api_key}"
    )

    session = requests.Session()

    completed: dict[int, list[float]] = {}
    cache_f = None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = _texts_fingerprint(texts)
        completed = _load_google_cache(cache_path, fingerprint, len(texts))
        if completed:
            log.info(
                "  resuming google embed: %d/%d already cached at %s",
                len(completed),
                len(texts),
                cache_path,
            )
            cache_f = open(cache_path, "a", encoding="utf-8")
        else:
            cache_f = open(cache_path, "w", encoding="utf-8")
            cache_f.write(
                json.dumps(
                    {"fingerprint": fingerprint, "n": len(texts), "model": model_name}
                )
                + "\n"
            )
            cache_f.flush()

    try:
        with tqdm(
            total=len(texts),
            desc="google embed",
            unit="txt",
            initial=len(completed),
        ) as pbar:
            for i, text in enumerate(texts):
                if i in completed:
                    continue
                emb: list[float] | None = None
                for attempt in range(max_retries):
                    try:
                        resp = session.post(
                            url,
                            json={"instances": [{"content": text}]},
                            timeout=30,
                        )
                        if resp.status_code == 429 or resp.status_code >= 500:
                            raise requests.HTTPError(
                                f"{resp.status_code} {resp.reason}", response=resp
                            )
                        resp.raise_for_status()
                        emb = resp.json()["predictions"][0]["embeddings"]["values"]
                        break
                    except Exception as e:
                        status = getattr(
                            getattr(e, "response", None), "status_code", None
                        )
                        if attempt == max_retries - 1:
                            log.error(
                                "  google failed at text %d after %d attempts — "
                                "partial progress saved to %s",
                                i,
                                max_retries,
                                cache_path,
                            )
                            raise
                        backoff = (2**attempt) * (2.0 if status == 429 else 1.0)
                        log.warning(
                            "  google retry %d/%d (text %d, status=%s): %s — sleeping %.1fs",
                            attempt + 1,
                            max_retries,
                            i,
                            status,
                            e,
                            backoff,
                        )
                        time.sleep(backoff)

                assert emb is not None
                completed[i] = emb
                if cache_f is not None:
                    cache_f.write(json.dumps({"idx": i, "emb": emb}) + "\n")
                    cache_f.flush()
                pbar.update(1)
                if sleep_between > 0:
                    time.sleep(sleep_between)
    finally:
        if cache_f is not None:
            cache_f.close()

    return np.asarray([completed[i] for i in range(len(texts))], dtype=np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    ids: list[str],
    labels: np.ndarray,
    path: Path,
    model_name: str,
    sub_exp: str,
    system: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=embeddings,
        ids=np.array(ids, dtype=object),
        labels=labels,
        metadata=np.array(
            json.dumps(
                {
                    "model_name": model_name,
                    "embedding_dim": embeddings.shape[1],
                    "sub_experiment": sub_exp,
                    "system": system,
                    "n_samples": len(ids),
                }
            )
        ),
    )
    log.info("  saved embeddings to %s (shape=%s)", path, embeddings.shape)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["ids"].tolist()


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


def cv_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
) -> dict:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        # 80/20 inner split for early stopping
        n_train = len(X_train_full)
        n_val = max(1, int(n_train * 0.2))
        inner_perm = np.random.RandomState(RANDOM_SEED + fold_idx).permutation(n_train)
        val_inner = inner_perm[:n_val]
        train_inner = inner_perm[n_val:]

        X_tr = X_train_full[train_inner]
        y_tr = y[train_idx][train_inner]
        X_vl = X_train_full[val_inner]
        y_vl = y[train_idx][val_inner]

        model = _train_mlp_fold(
            X_tr,
            y_tr,
            X_vl,
            y_vl,
            input_dim=X.shape[1],
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

        model.eval()
        with torch.no_grad():
            test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(test_t).cpu().numpy()
        proba = np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1)
        preds = np.argmax(logits, axis=1)

        folds.append(
            {
                "fold": fold_idx,
                "n_train": len(train_inner),
                "n_val": len(val_inner),
                "n_test": len(test_idx),
                "accuracy": float(accuracy_score(y[test_idx], preds)),
                "f1": float(f1_score(y[test_idx], preds, average="macro")),
                "auc": (
                    float(roc_auc_score(y[test_idx], proba))
                    if len(np.unique(y[test_idx])) > 1
                    else None
                ),
            }
        )
        log.info(
            "  MLP fold %d/5: acc=%.3f f1=%.3f auc=%s",
            fold_idx + 1,
            folds[-1]["accuracy"],
            folds[-1]["f1"],
            f"{folds[-1]['auc']:.3f}" if folds[-1]["auc"] is not None else "N/A",
        )

        del model

    return agg_folds(folds)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _embed_texts(
    texts: list[str],
    model_key: str,
    args,
    cache_path: Path | None = None,
) -> np.ndarray:
    cfg = EMBEDDING_MODELS[model_key]
    backend = cfg["backend"]
    model_name = cfg["model_name"]

    if backend == "llamacpp":
        return embed_with_llamacpp(
            texts,
            model_name=model_name,
            base_url=args.llamacpp_base_url,
            instruction=args.embedder_instruction,
        )
    elif backend == "google":
        return embed_with_google(
            texts,
            model_name=model_name,
            sleep_between=args.google_sleep_between,
            cache_path=cache_path,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_one(args, sub_exp: str, system: str) -> dict:
    cover_type = SUB_EXP_COVER[sub_exp]
    log.info("=== embeddings %s / %s (S vs %s) ===", sub_exp, system, cover_type)

    stego_recs, cover_recs = load_pair(args.data_dir, system, sub_exp)
    if not stego_recs or not cover_recs:
        log.warning("Missing data for %s/%s — skipping", sub_exp, system)
        return {}

    texts = [r["text"] for r in stego_recs] + [r["text"] for r in cover_recs]
    ids = [r["id"] for r in stego_recs] + [r["id"] for r in cover_recs]
    y = np.array([1] * len(stego_recs) + [0] * len(cover_recs))

    out_dir = stegoanalysis_dir(args.data_dir)
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    models_result: dict = {}

    for model_key in args.models:
        cfg = EMBEDDING_MODELS[model_key]
        slug = cfg["slug"]
        npz_path = emb_dir / f"emb_{sub_exp}_{system}_{slug}.npz"

        # Resume cache lives next to the final npz, only used for backends that
        # support it (currently google).
        cache_path = (
            npz_path.with_suffix(".partial.jsonl")
            if cfg["backend"] == "google"
            else None
        )

        # Embedding step
        if not args.skip_embed:
            if npz_path.exists():
                existing_X, _, existing_ids = load_embeddings(npz_path)
                if len(existing_ids) == len(ids):
                    log.info(
                        "  %s: embeddings already exist (%d records), skipping",
                        slug,
                        len(ids),
                    )
                else:
                    log.info(
                        "  %s: stale embeddings (%d != %d), re-generating",
                        slug,
                        len(existing_ids),
                        len(ids),
                    )
                    X = _embed_texts(texts, model_key, args, cache_path=cache_path)
                    save_embeddings(
                        X, ids, y, npz_path, cfg["model_name"], sub_exp, system
                    )
                    if cache_path is not None and cache_path.exists():
                        cache_path.unlink()
            else:
                log.info("  %s: generating embeddings for %d texts", slug, len(texts))
                X = _embed_texts(texts, model_key, args, cache_path=cache_path)
                save_embeddings(X, ids, y, npz_path, cfg["model_name"], sub_exp, system)
                if cache_path is not None and cache_path.exists():
                    cache_path.unlink()

        # Classification step
        if not args.skip_classify:
            if not npz_path.exists():
                log.warning(
                    "  %s: no embeddings at %s, skipping classification", slug, npz_path
                )
                models_result[slug] = {"error": "embeddings not found"}
                continue

            X, y_loaded, _ = load_embeddings(npz_path)
            log.info(
                "  %s: running MLP CV (dim=%d, n=%d)", slug, X.shape[1], X.shape[0]
            )

            mlp_result = cv_mlp(X, y_loaded)
            mlp_result["embedding_dim"] = int(X.shape[1])
            mlp_result["backend"] = cfg["backend"]
            mlp_result["classifier"] = "mlp"
            mlp_result["mlp_config"] = {
                "hidden1": 256,
                "hidden2": 64,
                "dropout": 0.3,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "max_epochs": 100,
                "patience": 10,
            }
            models_result[slug] = mlp_result

    result = {
        "sub_experiment": sub_exp,
        "system": system,
        "cover_type": cover_type,
        "n_stego": len(stego_recs),
        "n_cover": len(cover_recs),
        "seed": RANDOM_SEED,
        "models": models_result,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c — Signal 2: embedding generation + MLP classification"
    )
    add_common_args(parser)
    parser.add_argument(
        "--models",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(EMBEDDING_MODELS.keys()),
        help="Comma-separated model keys to run. Choices: qwen3, google",
    )
    parser.add_argument(
        "--llamacpp-base-url",
        default="http://127.0.0.1:11435/v1",
        help="Base URL for the llama.cpp embeddings server.",
    )
    parser.add_argument(
        "--embedder-instruction",
        default=DEFAULT_EMBEDDER_INSTRUCTION,
        help="Instruction prefix for Qwen3-Embedding-style models.",
    )
    parser.add_argument(
        "--google-sleep-between",
        type=float,
        default=0.1,
        help="Seconds to sleep between Google API requests (serial; partial "
        "progress is cached to a .partial.jsonl sidecar and auto-resumed).",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding generation; only run MLP on existing .npz files.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip MLP classification; only generate embeddings.",
    )
    args = parser.parse_args()

    for m in args.models:
        if m not in EMBEDDING_MODELS:
            parser.error(
                f"Unknown model key: {m}. Choices: {list(EMBEDDING_MODELS.keys())}"
            )

    seed_everything()
    out_dir = stegoanalysis_dir(args.data_dir)

    for sub_exp, system in iter_tasks(args):
        result = run_one(args, sub_exp, system)
        if not result:
            continue

        out_path = out_dir / f"embedding_mlp_{sub_exp}_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
