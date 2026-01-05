from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

OUT_DIR = Path("./src/pca/unit_test/artifacts")
OUT_DIR.mkdir(exist_ok=True)

EMBED_NPY = OUT_DIR / "embeddings.npy"

PCA_COMPONENTS = 5
EPS = 1e-8


def sanity_checks(X, k):
    n, d = X.shape
    print(f"Samples: {n}, Dim: {d}")

    # ---- Check NaNs / infs
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf")

    # ---- Variance check
    var_per_dim = X.var(axis=0)
    total_var = var_per_dim.sum()
    print(f"Total variance: {total_var:.4e}")

    if total_var < 1e-3:
        raise ValueError("Total variance too small — embeddings likely collapsed")

    # ---- Sample size check
    if n < 20 * k:
        print(
            f"WARNING: Only {n} samples for {k} PCA components (recommended ≥ {20 * k})"
        )

    # ---- Effective rank
    eigvals = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    eigvals = np.maximum(eigvals, 0)

    effective_rank = np.exp(
        -np.sum((eigvals / eigvals.sum()) * np.log((eigvals / eigvals.sum()) + EPS))
    )

    print(f"Effective rank (entropy-based): {effective_rank:.2f}")

    if effective_rank < k:
        print(f"WARNING: Effective rank < PCA_COMPONENTS ({effective_rank:.2f} < {k})")


def main():
    embeddings = np.load(EMBED_NPY)
    print(f"Loaded embeddings: {embeddings.shape}")

    mean = embeddings.mean(axis=0)
    X = embeddings - mean

    sanity_checks(X, PCA_COMPONENTS)

    # train pca
    pca = PCA(n_components=PCA_COMPONENTS)
    Z = pca.fit_transform(X)

    thresholds = np.median(Z, axis=0)
    np.save(OUT_DIR / "naive_pca_components.npy", pca.components_)
    np.save(OUT_DIR / "pca_components.npy", pca.components_)
    np.save(OUT_DIR / "pca_mean.npy", mean)
    np.save(OUT_DIR / "pca_thresholds.npy", thresholds)
    np.save(OUT_DIR / "pca_explained_variance.npy", pca.explained_variance_ratio_)

    print("PCA trained.")
    print("Explained variance:", pca.explained_variance_ratio_)


if __name__ == "__main__":
    main()
