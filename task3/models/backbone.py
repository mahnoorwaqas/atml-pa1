"""
Task 3 reuses Task 2's ResNet-18 + BatchNorm-freeze policy UNCHANGED, per
the assignment's "exactly the same ... ResNet-18 initialization ...
classifier head" requirement -- this re-exports task2.models.backbone
rather than duplicating it, so a fix to one automatically applies to both.
"""
from task2.models.backbone import build_resnet18, forward_features, set_bn_eval  # noqa: F401
