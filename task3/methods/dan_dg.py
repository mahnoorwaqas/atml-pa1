"""
Step 2: DAN-DG -- pairwise MMD alignment across the three OBSERVED source
domains only (never Sketch). Same MMD kernel construction as Task 2's DAN
(shared/mmd.py), applied to the 512-d feature before the classifier head.
lambda_dg=1 for the main comparison; Step 5's controlled study sweeps this
via configs/dan_dg.yaml's controlled_study section.
"""
import itertools

import torch.nn.functional as F

from shared.mmd import mmd_squared
from task3.models.backbone import build_resnet18, forward_features


def build_model(cfg):
    return build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])


def make_compute_loss(cfg):
    lambda_dg = cfg["method"]["lambda_dg"]
    bandwidth_scales = tuple(cfg["method"].get("bandwidth_scales", (0.5, 1.0, 2.0)))

    def compute_loss(model, source_images, source_labels, source_domain_ids, target_images, step_progress):
        # target_images is always None here -- Task 3 never builds a target loader.
        features = forward_features(model, source_images)
        logits = model.fc(features)
        cls_loss = F.cross_entropy(logits, source_labels)

        # Split features back out by domain (source_domain_ids in {0,1,2} for
        # Photo/ArtPainting/Cartoon, per pacs_protocol's iteration order) so
        # MMD can be computed over every UNORDERED pair of domains.
        unique_domains = source_domain_ids.unique().tolist()
        domain_feats = {d: features[source_domain_ids == d] for d in unique_domains}

        mmd_total = 0.0
        n_pairs = 0
        for d1, d2 in itertools.combinations(unique_domains, 2):
            mmd_total = mmd_total + mmd_squared(domain_feats[d1], domain_feats[d2], bandwidth_scales)
            n_pairs += 1
        mmd_loss = mmd_total / max(n_pairs, 1)

        total = cls_loss + lambda_dg * mmd_loss
        return total, {"cls_loss": float(cls_loss.item()), "mmd_loss": float(mmd_loss.item())}

    return compute_loss
