"""
Domain separability score (Step 5): freeze the trained backbone, collect
EQUAL numbers of source-validation and target features, 70/30 split (seed
6304), train a BALANCED logistic-regression classifier (C=1) to predict
source vs. target. Held-out accuracy is the score; 50% = chance.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task2.models.backbone import forward_features


@torch.no_grad()
def _extract_features(model, loader, device):
    model.eval()
    feats = []
    for images, _ in loader:
        images = images.to(device)
        feats.append(forward_features(model, images).cpu().numpy())
    return np.concatenate(feats, axis=0)


def domain_separability_score(model, source_val_loaders, target_loader, device, seed=6304):
    """
    Pools features across the 3 source validation domains, subsamples to
    match counts with the target domain (equal numbers per the assignment),
    then fits/evaluates the balanced logistic-regression probe.
    """
    source_feats = np.concatenate(
        [_extract_features(model, loader, device) for loader in source_val_loaders.values()],
        axis=0,
    )
    target_feats = _extract_features(model, target_loader, device)

    n = min(source_feats.shape[0], target_feats.shape[0])
    rng = np.random.RandomState(seed)
    source_idx = rng.choice(source_feats.shape[0], size=n, replace=False)
    target_idx = rng.choice(target_feats.shape[0], size=n, replace=False)

    X = np.concatenate([source_feats[source_idx], target_feats[target_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])  # 0 = source, 1 = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
