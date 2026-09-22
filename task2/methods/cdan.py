"""
Step 4: CDAN -- class-conditional adversarial alignment. Same discriminator
architecture/schedule/loss-weight as DANN, but conditioned on
g(x) = vec(f ⊗ p) instead of f alone (input_dim = feature_dim * n_classes).
Neither f nor p is detached, per the assignment's explicit requirement.
"""
import torch
import torch.nn.functional as F

from task2.models.backbone import build_resnet18, forward_features
from task2.models.domain_discriminator import DomainDiscriminator, grl_alpha, cdan_feature


def build_model(cfg):
    model = build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])
    input_dim = model.feature_dim * cfg["data"]["n_classes"]
    model.domain_discriminator = DomainDiscriminator(
        input_dim=input_dim,
        hidden_dim=cfg["method"].get("discriminator_hidden_dim", 256),
        dropout=cfg["method"].get("discriminator_dropout", 0.5),
    )
    return model


def make_compute_loss(cfg):
    max_grl_strength = cfg["method"].get("max_grl_strength", 1.0)

    def compute_loss(model, source_images, source_labels, source_domain_ids, target_images, step_progress):
        source_features = forward_features(model, source_images)
        target_features = forward_features(model, target_images)

        source_logits = model.fc(source_features)
        target_logits = model.fc(target_features)
        cls_loss = F.cross_entropy(source_logits, source_labels)

        # NOT detached, per the assignment -- gradients from the domain loss
        # flow back through both the features AND the class-probability path.
        source_probs = F.softmax(source_logits, dim=1)
        target_probs = F.softmax(target_logits, dim=1)

        source_g = cdan_feature(source_features, source_probs)
        target_g = cdan_feature(target_features, target_probs)

        alpha = grl_alpha(step_progress, max_strength=max_grl_strength)
        combined_g = torch.cat([source_g, target_g], dim=0)
        domain_labels = torch.cat([
            torch.zeros(source_g.shape[0], dtype=torch.long, device=source_g.device),
            torch.ones(target_g.shape[0], dtype=torch.long, device=target_g.device),
        ])
        domain_logits = model.domain_discriminator(combined_g, alpha=alpha)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total = cls_loss + domain_loss
        return total, {
            "cls_loss": float(cls_loss.item()),
            "domain_loss": float(domain_loss.item()),
            "alpha": alpha,
        }

    return compute_loss
