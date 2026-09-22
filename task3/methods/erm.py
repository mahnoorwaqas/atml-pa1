"""
Step 1: ERM baseline. NOT trained here -- reuses the Source-only checkpoint
already trained in Task 2 UNCHANGED, per the assignment ("load its saved
checkpoint rather than retraining it under a different configuration").
Exists only so train.py/evaluate_sketch.py can handle "erm" uniformly
alongside "dan_dg"/"sam" (same build_model signature), without a training loop.
"""
import os

import torch

from task3.models.backbone import build_resnet18

TASK2_SOURCE_ONLY_CHECKPOINT = "task2/results/checkpoints/source_only.pt"


def build_model(cfg):
    return build_resnet18(n_classes=cfg["data"]["n_classes"], weights=None)


def load_checkpoint(cfg, device):
    if not os.path.exists(TASK2_SOURCE_ONLY_CHECKPOINT):
        raise FileNotFoundError(
            f"{TASK2_SOURCE_ONLY_CHECKPOINT} not found -- train Task 2's "
            "source_only method first: python3 task2/train.py --config "
            "task2/configs/source_only.yaml"
        )
    model = build_model(cfg)
    state = torch.load(TASK2_SOURCE_ONLY_CHECKPOINT, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model
