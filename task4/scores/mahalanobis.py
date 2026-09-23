"""
u_Mah(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c)
Estimated from UNAUGMENTED CIFAR-10 training features: per-class means
mu_c, and ONE SHARED diagonal covariance Sigma (+1e-6 on every diagonal
entry), per the assignment.
"""
import numpy as np


def fit_mahalanobis_stats(train_features, train_labels, n_classes=10, eps=1e-6):
    """
    train_features: (N, D) from UNAUGMENTED training images.
    train_labels: (N,) integer class labels.
    Returns (class_means: (C, D), inv_diag_cov: (D,)) -- inv_diag_cov is the
    per-dimension 1/variance of the SHARED diagonal covariance, pooled
    across every class's centered residuals (not computed per-class), so
    distance reduces to a fast elementwise weighted sum rather than a full
    matrix inverse.
    """
    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)
    D = train_features.shape[1]

    class_means = np.zeros((n_classes, D))
    centered = np.zeros_like(train_features)
    for c in range(n_classes):
        mask = train_labels == c
        class_means[c] = train_features[mask].mean(axis=0)
        centered[mask] = train_features[mask] - class_means[c]

    diag_var = centered.var(axis=0) + eps
    inv_diag_cov = 1.0 / diag_var

    return class_means, inv_diag_cov


def mahalanobis_score(features, class_means, inv_diag_cov):
    """
    features: (N, D). Returns (N,) unknownness = min over classes of the
    diagonal Mahalanobis distance to that class's mean.
    """
    features = np.asarray(features)
    diffs = features[:, None, :] - class_means[None, :, :]  # (N, C, D)
    sq_dists = np.einsum("ncd,d,ncd->nc", diffs, inv_diag_cov, diffs)
    return sq_dists.min(axis=1)
