"""Single source of truth for on-disk data and PCA artifact locations.

The library, experiment drivers, and measurement scripts all read PCA-derived
data (reference corpora and trained PCA-hash artifacts) by path. Centralizing
those locations here keeps them out of scattered string literals and lets the
whole repo agree on where the ``pca/`` data tree lives.

Paths resolve relative to the repository root (this file is at
``src/embeddings/paths.py``), so they hold regardless of the current working
directory. Set the ``PCA_DIR`` environment variable to point the data root
somewhere other than ``<repo>/pca`` (e.g. a shared scratch volume).
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/embeddings/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Root of the PCA data/artifact tree (the former src/pca, now top-level pca/).
PCA_DIR = Path(os.environ.get("PCA_DIR", REPO_ROOT / "pca"))


def pca_artifacts_dir(domain: str) -> Path:
    """Directory of trained PCA-hash artifacts for ``domain``."""
    return PCA_DIR / domain / "artifacts"


def litreview_references() -> tuple[Path, Path]:
    """``(corpus.jsonl, references.jsonl)`` for the LitReview system corpus."""
    base = PCA_DIR / "litreview" / "references"
    return base / "corpus.jsonl", base / "references.jsonl"


def litreview_papers() -> Path:
    """Full paper-id pool used to expand LitReview prompts (``papers.jsonl``)."""
    return PCA_DIR / "litreview" / "references" / "papers.jsonl"
