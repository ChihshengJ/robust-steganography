from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

### Configuration

OUT_DIR = Path("./src/pca/summary/artifacts")
OUT_DIR.mkdir(exist_ok=True)

# Input files
EMBED_NPY = OUT_DIR / "fact_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "paraphrase_group_labels.npy"

# Output files
PCA_COMPONENTS_NPY = OUT_DIR / "pca_components.npy"
PCA_MEAN_NPY = OUT_DIR / "pca_mean.npy"
PCA_THRESHOLDS_NPY = OUT_DIR / "pca_thresholds.npy"
PCA_EXPLAINED_VAR_NPY = OUT_DIR / "pca_explained_variance.npy"
PCA_STABILITY_NPY = OUT_DIR / "pca_stability_scores.npy"
PCA_CONSISTENCY_NPY = OUT_DIR / "pca_consistency_rates.npy"

# Settings
PCA_COMPONENTS = 10
EPS = 1e-8


### Sanity Checks


def sanity_checks(X: np.ndarray, k: int):
    """Validate embeddings before PCA."""
    n, d = X.shape
    print(f"Samples: {n}, Dim: {d}")

    # NaN/Inf check
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or Inf")

    # Variance check
    var_per_dim = X.var(axis=0)
    total_var = var_per_dim.sum()
    print(f"Total variance: {total_var:.4e}")

    if total_var < 1e-3:
        raise ValueError("Total variance too small — embeddings likely collapsed")

    # Sample size check
    if n < 20 * k:
        print(f"WARNING: Only {n} samples for {k} components (recommended ≥ {20 * k})")

    # Effective rank
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    eigvals_norm = eigvals / (eigvals.sum() + EPS)
    effective_rank = np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm + EPS)))
    print(f"Effective rank (entropy-based): {effective_rank:.2f}")

    if effective_rank < k:
        print(f"WARNING: Effective rank < PCA_COMPONENTS ({effective_rank:.2f} < {k})")


### Paraphrase Stability Analysis


def compute_stability_scores(
    components: np.ndarray,
    mean: np.ndarray,
    original_emb: np.ndarray,
    para_emb: np.ndarray,
    group_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute stability score for each PCA component.

    Stability = between-group variance / within-group variance
    Higher = more stable under paraphrasing
    """
    print("\nComputing stability scores...")

    # Center embeddings
    all_emb = np.vstack([original_emb, para_emb])
    all_labels = np.concatenate([np.arange(len(original_emb)), group_labels])
    centered = all_emb - mean

    unique_labels = np.unique(all_labels)
    n_components = components.shape[0]
    embed_dim = components.shape[1]

    # Compute scatter matrices
    Sw = np.zeros((embed_dim, embed_dim))  # Within-group
    Sb = np.zeros((embed_dim, embed_dim))  # Between-group
    overall_mean = centered.mean(axis=0)

    for label in tqdm(unique_labels, desc="Scatter matrices", leave=False):
        mask = all_labels == label
        group = centered[mask]
        group_mean = group.mean(axis=0)

        # Within-group
        group_centered = group - group_mean
        Sw += group_centered.T @ group_centered

        # Between-group
        diff = (group_mean - overall_mean).reshape(-1, 1)
        Sb += len(group) * (diff @ diff.T)

    # Compute stability for each component
    stability_scores = []
    for i in range(n_components):
        direction = components[i]
        within_var = direction @ Sw @ direction
        between_var = direction @ Sb @ direction
        stability = between_var / (within_var + EPS)
        stability_scores.append(stability)

    return np.array(stability_scores)


def compute_consistency_rates(
    components: np.ndarray,
    mean: np.ndarray,
    thresholds: np.ndarray,
    original_emb: np.ndarray,
    para_emb: np.ndarray,
    group_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute hash consistency rate for each component.

    Consistency = fraction of paraphrases that hash to same bit as original.
    """
    print("\nComputing consistency rates...")

    n_components = components.shape[0]
    consistency_rates = []

    for i in range(n_components):
        direction = components[i]
        threshold = thresholds[i]

        # Hash originals
        orig_proj = (original_emb - mean) @ direction
        orig_bits = (orig_proj > threshold).astype(int)

        # Hash paraphrases
        para_proj = (para_emb - mean) @ direction
        para_bits = (para_proj > threshold).astype(int)

        # Check consistency
        matches = sum(
            para_bits[i] == orig_bits[group_labels[i]] for i in range(len(para_bits))
        )
        consistency = matches / len(para_bits)
        consistency_rates.append(consistency)

    return np.array(consistency_rates)


def print_component_analysis(
    explained_var: np.ndarray,
    stability_scores: np.ndarray | None,
    consistency_rates: np.ndarray | None,
    n_show: int = 10,
):
    """Print analysis of top components."""
    print("\n" + "=" * 70)
    print("Component Analysis")
    print("=" * 70)

    header = f"{'PC':<4} {'Var%':<8}"
    if stability_scores is not None:
        header += f"{'Stability':<12}"
    if consistency_rates is not None:
        header += f"{'Consistency':<12}"
    print(header)
    print("-" * 70)

    for i in range(min(n_show, len(explained_var))):
        row = f"{i:<4} {explained_var[i] * 100:>6.2f}%"
        if stability_scores is not None:
            row += f"  {stability_scores[i]:>10.4f}"
        if consistency_rates is not None:
            row += f"  {consistency_rates[i] * 100:>9.1f}%"
        print(row)

    # Recommendations
    print("\n" + "-" * 70)
    print("Recommendations:")

    if consistency_rates is not None:
        best_consistency = np.argmax(consistency_rates)
        print(
            f"  Highest consistency: PC{best_consistency} ({consistency_rates[best_consistency] * 100:.1f}%)"
        )

        # Find components with >70% consistency
        good_components = np.where(consistency_rates > 0.7)[0]
        if len(good_components) > 0:
            print(f"  Components with >70% consistency: {list(good_components)}")
        else:
            print(
                "  WARNING: No components achieve >70% consistency under paraphrasing"
            )

    if stability_scores is not None:
        best_stability = np.argmax(stability_scores)
        print(
            f"  Highest stability: PC{best_stability} (score={stability_scores[best_stability]:.4f})"
        )

    print("=" * 70)


def main():
    print("=" * 60)
    print("PCA Training")
    print("=" * 60)

    # Load embeddings
    embeddings = np.load(EMBED_NPY)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Check for paraphrase data
    has_paraphrases = PARAPHRASE_EMBED_NPY.exists() and GROUP_LABELS_NPY.exists()
    if has_paraphrases:
        para_emb = np.load(PARAPHRASE_EMBED_NPY)
        group_labels = np.load(GROUP_LABELS_NPY)
        print(f"Loaded paraphrase embeddings: {para_emb.shape}")
        print(f"Loaded group labels: {group_labels.shape}")
    else:
        para_emb = None
        group_labels = None
        print("No paraphrase data found — skipping stability analysis")

    # Center and validate
    mean = embeddings.mean(axis=0)
    X = embeddings - mean
    sanity_checks(X, PCA_COMPONENTS)

    # Train PCA
    print("\nTraining PCA...")
    pca = PCA(n_components=PCA_COMPONENTS)
    Z = pca.fit_transform(X)

    # Compute thresholds (median for balanced bits)
    thresholds = np.median(Z, axis=0)

    # Check bit balance
    bits = (Z > thresholds).astype(int)
    bit_means = bits.mean(axis=0)
    print(
        f"Bit balance (should be ~0.5): min={bit_means.min():.3f}, max={bit_means.max():.3f}"
    )

    # Save basic PCA artifacts
    np.save(PCA_COMPONENTS_NPY, pca.components_)
    np.save(PCA_MEAN_NPY, mean)
    np.save(PCA_THRESHOLDS_NPY, thresholds)
    np.save(PCA_EXPLAINED_VAR_NPY, pca.explained_variance_ratio_)

    print("\nExplained variance (first 10):")
    print(pca.explained_variance_ratio_[:10])
    print(
        f"Cumulative (10 components): {pca.explained_variance_ratio_[:10].sum() * 100:.1f}%"
    )

    # Paraphrase stability analysis
    stability_scores = None
    consistency_rates = None

    if has_paraphrases:
        stability_scores = compute_stability_scores(
            pca.components_, mean, embeddings, para_emb, group_labels
        )
        np.save(PCA_STABILITY_NPY, stability_scores)

        consistency_rates = compute_consistency_rates(
            pca.components_, mean, thresholds, embeddings, para_emb, group_labels
        )
        np.save(PCA_CONSISTENCY_NPY, consistency_rates)

    # Print analysis
    print_component_analysis(
        pca.explained_variance_ratio_,
        stability_scores,
        consistency_rates,
    )

    # Final recommendation
    print("\nFor PCAHash:")
    if consistency_rates is not None:
        best = int(np.argmax(consistency_rates))
        print(f"  hash_fn = PCAHash(pca_dir='{OUT_DIR}', start={best}, end={best + 1})")
    else:
        print(f"  hash_fn = PCAHash(pca_dir='{OUT_DIR}', start=0, end=1)")

    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
