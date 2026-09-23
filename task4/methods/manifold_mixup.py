"""
Manifold mixup for PROSER's data-placeholder objective: mixes penultimate
layer-2 feature maps from two DIFFERENT-class examples via lambda ~ Beta(2,2).
"""
import torch
from torch.distributions import Beta


def manifold_mixup(features, labels, beta_alpha=2.0, beta_beta=2.0, generator=None):
    """
    features: (N, C, H, W) -- output of resnet_cifar.forward_up_to_layer2.
    labels: (N,) int labels, used only to find a pairing permutation with
    y_i != y_j for every pair (the assignment requires this explicitly).
    Returns (mixed_features, lam (N,), perm (N,)).
    """
    N = features.shape[0]
    device = features.device

    perm = torch.randperm(N, device=device, generator=generator)
    # Retry any pairs that landed on the same class -- bounded retries;
    # with 10 classes and typical batch sizes this resolves in 1-2 passes.
    for _ in range(10):
        same_class = labels == labels[perm]
        if not same_class.any():
            break
        resample = torch.randperm(N, device=device, generator=generator)
        perm = torch.where(same_class, resample, perm)

    beta_dist = Beta(torch.tensor(float(beta_alpha)), torch.tensor(float(beta_beta)))
    lam = beta_dist.sample((N,)).to(device)

    lam_reshaped = lam.view(N, 1, 1, 1)
    mixed_features = lam_reshaped * features + (1 - lam_reshaped) * features[perm]

    return mixed_features, lam, perm
