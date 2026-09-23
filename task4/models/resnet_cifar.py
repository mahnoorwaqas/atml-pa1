"""
CIFAR-appropriate ResNet-18: the ImageNet 7x7 stride-2 first convolution is
replaced with a 3x3 stride-1 convolution, and the initial max-pool is
removed -- both per the assignment, since 32x32 CIFAR images would lose too
much spatial resolution through the ImageNet-sized stem. Operates on raw
32x32 images (no resize).

Exposes forward_up_to_layer2 / forward_from_layer2 as a split point for
PROSER's manifold mixup (mixes AFTER layer2, BEFORE layer3, per the
assignment).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet_cifar(n_classes=10, n_dummy_classifiers=0):
    """
    Random initialization (weights=None), per the assignment ("Train Vanilla
    and GCSC from random initialization"). If n_dummy_classifiers > 0,
    attaches model.dummy_fc -- used by PROSER; leave at 0 for Vanilla/GCSC.
    """
    model = resnet18(weights=None, num_classes=n_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.feature_dim = model.fc.in_features  # 512

    if n_dummy_classifiers > 0:
        model.dummy_fc = nn.Linear(model.feature_dim, n_dummy_classifiers)

    return model


def forward_features(model, x):
    """Full penultimate feature (512-d), used by Mahalanobis and PROSER eval."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)  # Identity for the CIFAR stem
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def forward_up_to_layer2(model, x):
    """Everything through layer2 -- the manifold-mixup split point."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    return x


def forward_from_layer2(model, h):
    """layer3 onward, from a (possibly mixed) layer-2 feature map to the 512-d penultimate feature."""
    x = model.layer3(h)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def logits_and_dummy(model, feature):
    """feature: (N, 512). Returns (known_logits (N, n_classes), dummy_logits (N, n_dummy) or None)."""
    known_logits = model.fc(feature)
    dummy_logits = model.dummy_fc(feature) if hasattr(model, "dummy_fc") else None
    return known_logits, dummy_logits
