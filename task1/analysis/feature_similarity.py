"""
Cosine stability of a backbone's representation between clean and
transformed images:

    I_T = (1/N) * sum_i [ f(x_i)^T f(T(x_i)) / (||f(x_i)|| * ||f(T(x_i))||) ]

Computed separately per backbone, per intervention in
{grayscale, cue_conflict, translation, patch_shuffle}.
"""
import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cosine_stability(clean_features, transformed_features):
    """
    clean_features, transformed_features: (N, D) arrays, paired by index
    (row i of each must correspond to the same underlying image).
    Returns scalar I_T (mean cosine similarity across the N paired examples).
    """
    clean = _to_numpy(clean_features)
    transformed = _to_numpy(transformed_features)

    assert clean.shape == transformed.shape, (
        f"Shape mismatch: clean {clean.shape} vs transformed {transformed.shape} "
        "-- these must be paired features for the same N images."
    )

    clean_norm = np.linalg.norm(clean, axis=1)
    transformed_norm = np.linalg.norm(transformed, axis=1)

    # Guard against a degenerate all-zero feature vector (norm 0), which would
    # otherwise divide by zero rather than surfacing as a real, visible bug.
    eps = 1e-12
    dot = np.sum(clean * transformed, axis=1)
    cosine_sim = dot / (clean_norm * transformed_norm + eps)

    return float(np.mean(cosine_sim))


def cosine_stability_per_backbone(features_by_backbone_and_condition):
    """
    features_by_backbone_and_condition: dict
        {backbone_name: {"clean": (N,D), <intervention>: (N,D), ...}}
    Returns: {backbone_name: {intervention: I_T}}
    (every key other than "clean" in the inner dict is treated as an
    intervention and compared against that backbone's "clean" features)
    """
    results = {}
    for backbone_name, condition_features in features_by_backbone_and_condition.items():
        if "clean" not in condition_features:
            raise KeyError(
                f"'{backbone_name}' is missing a 'clean' entry -- cosine "
                "stability needs a clean/transformed pair to compare."
            )
        clean = condition_features["clean"]
        results[backbone_name] = {
            intervention: cosine_stability(clean, feats)
            for intervention, feats in condition_features.items()
            if intervention != "clean"
        }
    return results