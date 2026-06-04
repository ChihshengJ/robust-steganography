"""Single source of truth for on-disk data and PCA artifact locations.

The library, experiment drivers, and measurement scripts all read shared data by
path: the LitReview reference corpus (a shipped dataset under ``data/litreview/``)
and the legacy trained PCA-hash artifacts (under ``pca/``). Centralizing those
locations here keeps them out of scattered string literals and lets the whole repo
agree on where each data tree lives.

Paths resolve relative to the repository root (this file is at
``src/systems/paths.py``), so they hold regardless of the current working
directory. Set the ``DATA_DIR`` or ``PCA_DIR`` environment variable to point the
respective data root somewhere other than ``<repo>/data`` / ``<repo>/pca`` (e.g. a
shared scratch volume).
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/systems/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Root of the shipped data tree (reference corpora, experiment outputs).
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
LITREVIEW_DIR = DATA_DIR / "litreview"

# Root of the legacy PCA data/artifact tree (the former src/pca, now top-level pca/).
PCA_DIR = Path(os.environ.get("PCA_DIR", REPO_ROOT / "pca"))


def pca_artifacts_dir(domain: str) -> Path:
    """Directory of trained PCA-hash artifacts for ``domain`` (legacy)."""
    return PCA_DIR / domain / "artifacts"


def litreview_references() -> tuple[Path, Path]:
    """``(corpus.jsonl, references.jsonl)`` for the LitReview system corpus."""
    base = LITREVIEW_DIR / "references"
    return base / "corpus.jsonl", base / "references.jsonl"


def litreview_papers() -> Path:
    """Full paper-id pool used to expand LitReview prompts (``papers.jsonl``)."""
    return LITREVIEW_DIR / "references" / "papers.jsonl"
