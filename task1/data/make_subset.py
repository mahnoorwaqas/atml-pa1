"""
Builds:
  1. Stratified 80/20 train/val split from the official training partition (seed 6304).
  2. Class-balanced 500-image subset from the official test partition (seed 6304),
     with image identifiers saved so every downstream script reuses the exact same images.

Usage:
    python task1/data/make_subset.py --config task1/configs/config.yaml
"""
import argparse
import json
import random

import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit


def _get_labels(dataset):
    """
    Extract integer labels for every example in a torchvision dataset without
    decoding any images (both STL10 and OxfordIIITPet expose labels as a plain
    attribute, so this is cheap even for the training split).
    """
    if hasattr(dataset, "labels"):        # torchvision STL10
        return np.asarray(dataset.labels)
    if hasattr(dataset, "_labels"):       # torchvision OxfordIIITPet (>=0.13)
        return np.asarray(dataset._labels)
    if hasattr(dataset, "targets"):       # generic torchvision fallback
        return np.asarray(dataset.targets)
    raise AttributeError(
        "Could not find labels on this dataset object — check your torchvision "
        "version, since the internal attribute name for labels has changed "
        "across releases."
    )


def load_dataset(cfg):
    """
    Load STL-10 or Oxford-IIIT Pets via torchvision.
    Returns (train_dataset, test_dataset, class_names).
    Only downloads/reads metadata here — actual image tensors are loaded later,
    per-batch, by the backbone/eval pipeline.
    """
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]

    if name == "stl10":
        from torchvision.datasets import STL10
        # STL10's "train"/"test" splits are the labeled partitions; the
        # separate "unlabeled" split is intentionally not used here.
        train_dataset = STL10(root=root, split="train", download=True)
        test_dataset = STL10(root=root, split="test", download=True)
        class_names = list(train_dataset.classes)

    elif name == "oxford_pets":
        from torchvision.datasets import OxfordIIITPet
        # "trainval" is Oxford-IIIT Pets' official training partition.
        train_dataset = OxfordIIITPet(
            root=root, split="trainval", target_types="category", download=True
        )
        test_dataset = OxfordIIITPet(
            root=root, split="test", target_types="category", download=True
        )
        class_names = list(train_dataset.classes)

    else:
        raise ValueError(f"Unknown dataset '{name}' — expected 'stl10' or 'oxford_pets'.")

    return train_dataset, test_dataset, class_names


def stratified_split(train_dataset, split_ratio, seed):
    """
    Stratified 80/20 split of the official training partition.
    Returns (train_indices, val_indices) as plain Python lists of ints.
    """
    labels = _get_labels(train_dataset)
    indices = np.arange(len(labels))

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=split_ratio, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(indices, labels))

    return train_idx.tolist(), val_idx.tolist()


def class_balanced_subset(test_dataset, n_total, seed):
    """
    Select a class-balanced subset of n_total images from the test set.
    If a class has too few examples, use all available and document the
    imbalance (per_class_counts records requested/available/selected per class,
    so any shortfall — and the resulting final total, if it falls below
    n_total — is visible in the saved output).
    Returns (selected_indices, per_class_counts).
    """
    labels = _get_labels(test_dataset)
    classes = sorted(set(int(c) for c in labels.tolist()))
    n_classes = len(classes)

    target_per_class = n_total // n_classes
    remainder = n_total - target_per_class * n_classes  # spread 1 extra across the first `remainder` classes

    rng = random.Random(seed)
    selected_indices = []
    per_class_counts = {}

    for i, c in enumerate(classes):
        class_indices = np.where(labels == c)[0].tolist()
        rng.shuffle(class_indices)

        take = target_per_class + (1 if i < remainder else 0)
        chosen = class_indices[:take]  # naturally == all available if len < take

        per_class_counts[str(c)] = {
            "requested": take,
            "available": len(class_indices),
            "selected": len(chosen),
        }
        selected_indices.extend(chosen)

    return selected_indices, per_class_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]

    train_dataset, test_dataset, class_names = load_dataset(cfg)
    train_idx, val_idx = stratified_split(
        train_dataset, cfg["dataset"]["train_val_split"], seed
    )
    test_idx, per_class_counts = class_balanced_subset(
        test_dataset, cfg["subset"]["n_test_images"], seed
    )

    n_selected_total = sum(v["selected"] for v in per_class_counts.values())
    if n_selected_total < cfg["subset"]["n_test_images"]:
        print(
            f"WARNING: only selected {n_selected_total} of "
            f"{cfg['subset']['n_test_images']} requested test images — "
            f"one or more classes had insufficient examples. See per_class_counts."
        )

    out = {
        "seed": seed,
        "dataset": cfg["dataset"]["name"],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_subset_indices": test_idx,
        "per_class_counts": per_class_counts,
        "class_names": class_names,
    }
    with open(cfg["subset"]["ids_path"], "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved subset/split indices to {cfg['subset']['ids_path']}")
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, test subset: {n_selected_total}")


if __name__ == "__main__":
    main()