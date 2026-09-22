"""
Gradient Reversal Layer (GRL) + domain discriminator, shared by DANN and
CDAN (input_dim differs: 512 for DANN (just the feature f), 512*n_classes
for CDAN (the outer product f⊗p flattened) -- pass the right input_dim).
"""
import math

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha):
    return _GradientReversalFunction.apply(x, alpha)


def grl_alpha(progress, max_strength=1.0):
    """
    progress in [0, 1] (training progress, e.g. step / total_steps).
    Standard DANN schedule: alpha(p) = 2 / (1 + exp(-10p)) - 1, scaled by
    max_strength (used by Step 6's controlled study, which varies the
    MAXIMUM gradient-reversal strength over {0.25, 0.5, 1} while keeping
    this same schedule SHAPE).
    """
    base = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return max_strength * base


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output -- per the
    assignment's exact spec, shared by DANN (input_dim=feature_dim) and
    CDAN (input_dim=feature_dim * n_classes)."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, alpha=1.0):
        x = gradient_reversal(x, alpha)
        return self.net(x)


def cdan_feature(features, class_probs):
    """
    g(x) = vec(f ⊗ p): outer product of the (N, D) feature batch and the
    (N, C) softmax class-probability batch, flattened per-example to
    (N, D*C). Do NOT detach features or class_probs (the assignment
    explicitly forbids detaching either for the required implementation).
    """
    N, D = features.shape
    _, C = class_probs.shape
    outer = features.unsqueeze(2) * class_probs.unsqueeze(1)  # (N, D, C)
    return outer.reshape(N, D * C)
