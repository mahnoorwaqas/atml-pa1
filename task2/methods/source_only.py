"""
Step 1: Source-only ERM. Cross-entropy over the three labeled source
domains only (domain-balanced batches); no target data used at all. Save
this checkpoint and reuse it UNCHANGED as Task 3's ERM baseline -- don't
retrain it there.
"""
import torch.nn.functional as F

from task2.models.backbone import build_resnet18


def build_model(cfg):
    return build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])


def make_compute_loss(cfg):
    def compute_loss(model, source_images, source_labels, source_domain_ids, target_images, step_progress):
        logits = model(source_images)
        loss = F.cross_entropy(logits, source_labels)
        return loss, {"cls_loss": float(loss.item())}

    return compute_loss
