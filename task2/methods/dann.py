"""
Step 3: DANN -- adversarial domain alignment via a gradient-reversal layer
and a binary domain discriminator on the 512-d feature. Only source images
contribute to the classification loss; both source and target contribute to
the domain loss. The discriminator's parameters are trained by the SAME
optimizer step as the backbone (GRL already flips the sign of the gradient
flowing back into the backbone, so no separate discriminator optimizer or
extra backward pass is needed).
"""
import torch
import torch.nn.functional as F

from task2.models.backbone import build_resnet18, forward_features
from task2.models.domain_discriminator import DomainDiscriminator, grl_alpha


def build_model(cfg):
    model = build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])
    model.domain_discriminator = DomainDiscriminator(
        input_dim=model.feature_dim,
        hidden_dim=cfg["method"].get("discriminator_hidden_dim", 256),
        dropout=cfg["method"].get("discriminator_dropout", 0.5),
    )
    return model


def make_compute_loss(cfg):
    max_grl_strength = cfg["method"].get("max_grl_strength", 1.0)

    def compute_loss(model, source_images, source_labels, source_domain_ids, target_images, step_progress):
        source_features = forward_features(model, source_images)
        target_features = forward_features(model, target_images)

        cls_logits = model.fc(source_features)
        cls_loss = F.cross_entropy(cls_logits, source_labels)

        alpha = grl_alpha(step_progress, max_strength=max_grl_strength)
        combined_features = torch.cat([source_features, target_features], dim=0)
        domain_labels = torch.cat([
            torch.zeros(source_features.shape[0], dtype=torch.long, device=source_features.device),
            torch.ones(target_features.shape[0], dtype=torch.long, device=target_features.device),
        ])
        domain_logits = model.domain_discriminator(combined_features, alpha=alpha)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total = cls_loss + domain_loss  # unit weight, per the assignment
        return total, {
            "cls_loss": float(cls_loss.item()),
            "domain_loss": float(domain_loss.item()),
            "alpha": alpha,
        }

    return compute_loss
