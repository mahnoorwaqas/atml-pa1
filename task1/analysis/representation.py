"""
2D projection of clean + transformed features for each backbone, fit jointly
so both conditions share one embedding space.

Rules from the assignment:
  - One 2D projection per backbone, fit on COMBINED clean + transformed features.
  - Color = ground-truth class, marker style = clean vs. transformed.
  - Do NOT compare absolute coordinates across backbones (separately fit spaces).
  - Report the visualization settings used (perplexity / n_neighbors, etc).
"""
import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def fit_projection(combined_features, method="tsne", seed=6304, **kwargs):
    """
    combined_features: (N_clean + N_transformed, D) array for ONE backbone,
    ONE intervention.
    method: "tsne" or "umap".
    kwargs: passed through to the underlying method (e.g. perplexity=30 for
    t-SNE, n_neighbors=15 for UMAP — take these from
    config["representation_analysis"]).
    Returns (N, 2) projected coordinates, in the same row order as input.
    """
    X = _to_numpy(combined_features)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = kwargs.get("perplexity", 30)
        # perplexity must be < n_samples, per sklearn's constraint -- clamp
        # rather than let a small combined set (e.g. a handful of examples)
        # crash outright.
        perplexity = min(perplexity, max(X.shape[0] - 1, 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
        return reducer.fit_transform(X)

    elif method == "umap":
        import umap  # pip install umap-learn

        n_neighbors = kwargs.get("n_neighbors", 15)
        n_neighbors = min(n_neighbors, max(X.shape[0] - 1, 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed)
        return reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown method '{method}' -- expected 'tsne' or 'umap'.")


def plot_projection(coords_2d, class_labels, condition_labels, title, save_path):
    """
    Scatter plot: color by class_labels, marker style by condition_labels
    (e.g. "clean" vs. the intervention name). Saves to save_path.
    """
    import matplotlib.pyplot as plt

    coords_2d = _to_numpy(coords_2d)
    class_labels = _to_numpy(class_labels)
    condition_labels = np.asarray(condition_labels)

    unique_classes = sorted(set(class_labels.tolist()))
    unique_conditions = sorted(set(condition_labels.tolist()))

    if len(unique_conditions) > 2:
        raise ValueError(
            f"Expected exactly 2 conditions (clean vs. one intervention), got "
            f"{unique_conditions} -- plot one intervention at a time."
        )

    # Fixed marker per condition so "clean" always reads the same way across
    # every plot you generate, regardless of which intervention it's paired with.
    marker_by_condition = {}
    for cond in unique_conditions:
        marker_by_condition[cond] = "o" if cond == "clean" else "x"

    cmap = plt.get_cmap("tab10" if len(unique_classes) <= 10 else "tab20")
    color_by_class = {c: cmap(i % cmap.N) for i, c in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in unique_conditions:
        for cls in unique_classes:
            mask = (condition_labels == cond) & (class_labels == cls)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[color_by_class[cls]],
                marker=marker_by_condition[cond],
                s=20, alpha=0.7,
                label=f"class {cls} ({cond})",
            )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    # Full per-(class, condition) legend gets unwieldy fast; a compact
    # condition-only legend (marker shape meaning) plus the class colors
    # described in the caption is more readable in an 8-page report.
    condition_handles = [
        plt.Line2D([0], [0], marker=marker_by_condition[c], color="gray",
                   linestyle="", label=c)
        for c in unique_conditions
    ]
    ax.legend(handles=condition_handles, title="condition", loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)