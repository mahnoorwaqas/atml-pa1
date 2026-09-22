"""
Shared training/eval protocol for Tasks 2 and 3: stratified 80/20 splits per
source domain (seed 6304), domain-balanced batch sampling (8 per source
domain + 24 target for Task 2; 8 per source domain, no target, for Task 3),
and the transforms specified by the assignment (256x256 resize -> random 224
crop + hflip for train, 224 center crop for val/eval).
"""
import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvt

from shared.pacs import load_all_source_domains, load_target_domain

SPLIT_PATH = os.path.join(os.path.dirname(__file__), "splits", "pacs_sketch_seed6304.json")

# ResNet18_Weights.IMAGENET1K_V1's associated normalization.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = tvt.Compose([
    tvt.Resize((256, 256)),
    tvt.RandomCrop(224),
    tvt.RandomHorizontalFlip(),
    tvt.ToTensor(),
    tvt.Normalize(mean=_MEAN, std=_STD),
])
EVAL_TRANSFORM = tvt.Compose([
    tvt.Resize((256, 256)),
    tvt.CenterCrop(224),
    tvt.ToTensor(),
    tvt.Normalize(mean=_MEAN, std=_STD),
])


def build_source_splits(root, seed=6304):
    """
    Stratified 80/20 train/val split PER source domain. Saves to
    shared/splits/pacs_sketch_seed6304.json on first call; every later call
    (Task 2 or Task 3) loads the SAME split from disk rather than
    recomputing it, per the assignment's "reuse the same source splits
    across both tasks."
    """
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            return json.load(f)

    domains = load_all_source_domains(root)  # no transform needed -- only .targets used here
    splits = {}
    for domain_name, dataset in domains.items():
        labels = np.array(dataset.targets)
        indices = np.arange(len(labels))
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, labels))
        splits[domain_name] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        }

    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def default_steps_per_epoch(splits, per_source=8):
    """
    The assignment specifies "at most 30 source epochs" but doesn't pin down
    what one "epoch" means for a domain-balanced-batch protocol drawing from
    3 differently-sized source domains. This defines one epoch as enough
    steps to see the SMALLEST source domain's training split once -- a
    common convention for multi-domain balanced sampling. State this choice
    in your report if you keep it; override training.steps_per_epoch in
    base.yaml to use a fixed value instead.
    """
    min_size = min(len(v["train_indices"]) for v in splits.values())
    return max(1, min_size // per_source)


class _Subset(Dataset):
    """Wraps an ImageFolder + a fixed index list + a transform. Keeping the
    transform here (not baked into the ImageFolder itself) lets the SAME
    underlying ImageFolder serve both train and eval subsets without
    reloading images from disk twice."""

    def __init__(self, image_folder, indices, transform):
        self.image_folder = image_folder
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.image_folder[self.indices[i]]
        return self.transform(img), label


class DomainBalancedBatchIterator:
    """
    Yields batches shaped per the assignment: `per_source` examples from
    EACH source domain (concatenated), plus `per_target` target examples if
    a target_loader is given (Task 2) -- omit target_loader for Task 3 (no
    target access at all). Cycles any loader that runs out early, so this
    can be iterated for a fixed number of steps regardless of domain size.
    """

    def __init__(self, source_loaders, target_loader=None):
        self.source_loaders = source_loaders  # dict: domain_name -> DataLoader
        self.target_loader = target_loader
        self._source_iters = {d: iter(l) for d, l in source_loaders.items()}
        self._target_iter = iter(target_loader) if target_loader is not None else None

    def next_batch(self):
        source_images, source_labels, source_domain_ids = [], [], []
        for domain_idx, (domain_name, loader) in enumerate(self.source_loaders.items()):
            try:
                images, labels = next(self._source_iters[domain_name])
            except StopIteration:
                self._source_iters[domain_name] = iter(loader)
                images, labels = next(self._source_iters[domain_name])
            source_images.append(images)
            source_labels.append(labels)
            source_domain_ids.append(torch.full((images.shape[0],), domain_idx, dtype=torch.long))

        source_images = torch.cat(source_images, dim=0)
        source_labels = torch.cat(source_labels, dim=0)
        source_domain_ids = torch.cat(source_domain_ids, dim=0)

        if self.target_loader is None:
            return source_images, source_labels, source_domain_ids, None

        try:
            target_images, _ = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_loader)
            target_images, _ = next(self._target_iter)

        return source_images, source_labels, source_domain_ids, target_images


def make_domain_balanced_loaders(root, splits, per_source=8, per_target=24, include_target=True, seed=6304):
    """
    Builds one DataLoader per source domain (TRAIN_TRANSFORM, train split)
    and, if include_target, one target DataLoader (TRAIN_TRANSFORM, full
    target domain -- images used unlabeled, so labels are simply ignored
    downstream). Returns a DomainBalancedBatchIterator.
    """
    source_domains = load_all_source_domains(root)
    source_loaders = {}
    g = torch.Generator().manual_seed(seed)
    for domain_name, image_folder in source_domains.items():
        train_idx = splits[domain_name]["train_indices"]
        subset = _Subset(image_folder, train_idx, TRAIN_TRANSFORM)
        source_loaders[domain_name] = DataLoader(
            subset, batch_size=per_source, shuffle=True, drop_last=True, generator=g
        )

    target_loader = None
    if include_target:
        target_folder = load_target_domain(root)
        target_subset = _Subset(target_folder, list(range(len(target_folder))), TRAIN_TRANSFORM)
        target_loader = DataLoader(
            target_subset, batch_size=per_target, shuffle=True, drop_last=True, generator=g
        )

    return DomainBalancedBatchIterator(source_loaders, target_loader)


def make_eval_loader(image_folder, indices=None, batch_size=64):
    """Eval-time loader: EVAL_TRANSFORM (center crop, no augmentation)."""
    if indices is None:
        indices = list(range(len(image_folder)))
    subset = _Subset(image_folder, indices, EVAL_TRANSFORM)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
