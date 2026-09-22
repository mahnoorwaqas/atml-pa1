"""
ResNet-18 backbone + 7-class head, fine-tuned end-to-end (per the assignment
-- unlike Task 1, this is NOT a frozen-backbone/linear-probe setup).

BatchNorm policy (critical, and easy to get wrong): freeze all BatchNorm
RUNNING STATISTICS at their pretrained ImageNet values for every method in
Tasks 2 and 3, while keeping BatchNorm's learnable scale/bias (gamma/beta)
trainable. Call model.train() as usual, then switch every BatchNorm
submodule back to eval() -- NOT the whole model -- so running_mean/
running_var stop updating while their affine parameters still get gradients
like any other trainable parameter.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(n_classes=7, weights="IMAGENET1K_V1"):
    """
    Returns a full ResNet-18 with its final fc replaced by an n_classes
    linear layer. model.feature_dim (512) is attached for convenience
    (used to size the domain discriminator in DANN/CDAN).
    """
    weights_enum = getattr(ResNet18_Weights, weights) if weights is not None else None
    model = resnet18(weights=weights_enum)
    feature_dim = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(feature_dim, n_classes)
    model.feature_dim = feature_dim
    return model


def forward_features(model, x):
    """
    Runs everything up to (not including) model.fc, returning the
    512-dimensional pooled feature -- what DAN/DANN/CDAN's alignment losses
    operate on, and what model.fc then consumes for classification.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def set_bn_eval(model):
    """
    Call this AFTER model.train() on every training step. Puts every
    BatchNorm2d submodule (and ONLY those) into eval mode, so running
    statistics stop updating while weight/bias remain trainable as usual.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
