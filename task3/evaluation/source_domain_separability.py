"""
Source-domain separability score (Step 4): freeze the backbone, collect
BALANCED features from the three source validation sets, 70/30 split (seed
6304), train a MULTINOMIAL logistic-regression classifier (C=1) to predict
which of the 3 source domains each feature came from. Held-out accuracy is
the score; chance = 33.3% (vs. Task 2's binary source-vs-target score).
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task3.models.backbone import forward_features


@torch.no_grad()
def _extract_features(model, loader, device):
    model.eval()
    feats = []
    for images, _ in loader:
        images = images.to(device)
        feats.append(forward_features(model, images).cpu().numpy())
    return np.concatenate(feats, axis=0)


def source_domain_separability_score(model, source_val_loaders, device, seed=6304):
    domain_names = list(source_val_loaders.keys())
    feats_by_domain = {
        name: _extract_features(model, loader, device)
        for name, loader in source_val_loaders.items()
    }

    n = min(f.shape[0] for f in feats_by_domain.values())  # equal numbers per domain
    rng = np.random.RandomState(seed)

    X_parts, y_parts = [], []
    for domain_idx, name in enumerate(domain_names):
        feats = feats_by_domain[name]
        idx = rng.choice(feats.shape[0], size=n, replace=False)
        X_parts.append(feats[idx])
        y_parts.append(np.full(n, domain_idx))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, max_iter=1000)  # multinomial by default for >2 classes (lbfgs solver)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
