"""
The classifier head here is just `model.fc` on the ResNet-18 built by
backbone.build_resnet18() -- Task 2 fine-tunes the WHOLE network end-to-end
(unlike Task 1's frozen-backbone + separate linear head), so there's no
separate head module to instantiate. This file exposes small helpers so
other code doesn't need to know "the head" is just `.fc`.
"""
import torch.nn.functional as F


def classify(model, features_or_images, from_features=False):
    """
    If from_features=True, `features_or_images` is a precomputed (N, 512)
    feature batch (e.g. from backbone.forward_features) -- pass it straight
    through model.fc. Otherwise, run the FULL model forward pass (features
    + head) on raw images.
    """
    if from_features:
        return model.fc(features_or_images)
    return model(features_or_images)


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)
