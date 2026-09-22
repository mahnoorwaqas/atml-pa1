"""
Step 2: DAN -- MMD alignment on the 512-d feature immediately before the
classifier head. lambda_mmd=1 for the main comparison (Step 6's controlled
study sweeps this via configs/dan.yaml's controlled_study section).
"""
import torch.nn.functional as F

from shared.mmd import mmd_squared
from task2.models.backbone import build_resnet18, forward_features


def build_model(cfg):
    return build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])


def make_compute_loss(cfg):
    lambda_mmd = cfg["method"]["lambda_mmd"]
    bandwidth_scales = tuple(cfg["method"].get("bandwidth_scales", (0.5, 1.0, 2.0)))

    def compute_loss(model, source_images, source_labels, source_domain_ids, target_images, step_progress):
        source_features = forward_features(model, source_images)
        target_features = forward_features(model, target_images)

        cls_logits = model.fc(source_features)
        cls_loss = F.cross_entropy(cls_logits, source_labels)

        mmd_loss = mmd_squared(source_features, target_features, bandwidth_scales)

        total = cls_loss + lambda_mmd * mmd_loss
        return total, {"cls_loss": float(cls_loss.item()), "mmd_loss": float(mmd_loss.item())}

    return compute_loss
