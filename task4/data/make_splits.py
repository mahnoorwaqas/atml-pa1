"""
Stratified 90/10 split of CIFAR-10's official training partition, seed 6304.
Saved once and reused so every method (Vanilla, GCSC, PROSER) trains/selects
checkpoints against the IDENTICAL split.

Usage:
    python3 task4/data/make_splits.py --root ./data_raw
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from task4.data.hf_cifar import CIFAR10HF as CIFAR10

SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10_split_seed6304.json")


def build_cifar10_split(root, seed=6304):
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            return json.load(f)

    train_dataset = CIFAR10(root=root, train=True, download=True)
    labels = np.array(train_dataset.targets)
    indices = np.arange(len(labels))

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels))

    split = {"train_indices": train_idx.tolist(), "val_indices": val_idx.tolist()}
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data_raw")
    args = parser.parse_args()

    split = build_cifar10_split(args.root)
    print(f"train: {len(split['train_indices'])}, val: {len(split['val_indices'])}")
    print(f"Saved to {SPLIT_PATH}")
