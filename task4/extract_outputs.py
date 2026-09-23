"""
Extracts and caches penultimate features + logits for a trained checkpoint,
on: CIFAR-10 training (UNAUGMENTED -- for Mahalanobis stats), validation
(threshold calibration), test (final known-class evaluation), and the fixed
CIFAR-100 near/far unknowns. Every score in scores/ consumes these same
cached outputs, per the assignment ("All four scores must use exactly the
same saved logits and features").

Note: for the near/far splits, the cached "labels" are CIFAR-100 FINE
labels (which unknown class each image came from), not CIFAR-10 labels --
never used as a classification target, only for failure-analysis lookups.

Run from the REPO ROOT, after training each method:
    python3 task4/extract_outputs.py --config task4/configs/vanilla.yaml
    python3 task4/extract_outputs.py --config task4/configs/gcsc.yaml
    python3 task4/extract_outputs.py --config task4/configs/proser.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from tqdm import tqdm

from task4.data.cifar10 import load_cifar10, make_loader, EVAL_TRANSFORM, PLAIN_TRANSFORM
from task4.data.cifar100_unknowns import load_cifar100_unknowns, make_loader as make_unknown_loader
from task4.data.make_splits import build_cifar10_split
from task4.models.resnet_cifar import build_resnet_cifar, forward_features, logits_and_dummy


def _device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def _extract(model, loader, device, has_dummy=False):
    model.eval()
    all_feats, all_logits, all_dummy_logits, all_labels = [], [], [], []
    for images, labels in tqdm(loader, desc="extracting", leave=False):
        images = images.to(device)
        feats = forward_features(model, images)
        known_logits, dummy_logits = logits_and_dummy(model, feats)
        all_feats.append(feats.cpu())
        all_logits.append(known_logits.cpu())
        all_labels.append(labels)
        if has_dummy and dummy_logits is not None:
            all_dummy_logits.append(dummy_logits.cpu())

    out = {
        "features": torch.cat(all_feats),
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
    }
    if all_dummy_logits:
        out["dummy_logits"] = torch.cat(all_dummy_logits)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device()
    method_name = cfg["method"]["name"]
    has_dummy = method_name == "proser"
    n_dummy = cfg["method"].get("n_dummy", 5) if has_dummy else 0

    model = build_resnet_cifar(n_classes=cfg["data"]["n_classes"], n_dummy_classifiers=n_dummy).to(device)
    state = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    split = build_cifar10_split(cfg["data"]["root"], seed=cfg["seed"])
    train_dataset, test_dataset = load_cifar10(cfg["data"]["root"])
    near_dataset, far_dataset = load_cifar100_unknowns(cfg["data"]["root"])

    train_loader = make_loader(train_dataset, split["train_indices"], PLAIN_TRANSFORM, batch_size=256)
    val_loader = make_loader(train_dataset, split["val_indices"], EVAL_TRANSFORM, batch_size=256)
    test_loader = make_loader(test_dataset, list(range(len(test_dataset))), EVAL_TRANSFORM, batch_size=256)
    near_loader = make_unknown_loader(near_dataset, batch_size=256)
    far_loader = make_unknown_loader(far_dataset, batch_size=256)

    outputs = {
        "train": _extract(model, train_loader, device, has_dummy=has_dummy),
        "val": _extract(model, val_loader, device, has_dummy=has_dummy),
        "test": _extract(model, test_loader, device, has_dummy=has_dummy),
        "near": _extract(model, near_loader, device, has_dummy=has_dummy),
        "far": _extract(model, far_loader, device, has_dummy=has_dummy),
    }

    cache_dir = os.path.join("task4", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{method_name}_outputs.pt")
    torch.save(outputs, out_path)
    print(f"Saved cached outputs to {out_path}")


if __name__ == "__main__":
    main()
