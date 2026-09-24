"""
CIFAR-10 loading, transforms, and split-aware DataLoaders.
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvt
from task4.data.hf_cifar import CIFAR10HF as CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_train_transform(extra_transform=None):
    """
    Base: random 32x32 crop with 4px padding + random horizontal flip, per
    the assignment. extra_transform (e.g. RandAugment for GCSC) is inserted
    AFTER crop/flip and BEFORE ToTensor/Normalize, per the assignment's
    exact wording ("insert RandAugment ... after the crop and flip and
    before conversion and normalization").
    """
    ops = [tvt.RandomCrop(32, padding=4), tvt.RandomHorizontalFlip()]
    if extra_transform is not None:
        ops.append(extra_transform)
    ops += [tvt.ToTensor(), tvt.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return tvt.Compose(ops)


# No crop/flip -- used for validation/test evaluation AND for estimating
# Mahalanobis statistics from "unaugmented CIFAR-10 training features", per
# the assignment.
EVAL_TRANSFORM = tvt.Compose([tvt.ToTensor(), tvt.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
PLAIN_TRANSFORM = EVAL_TRANSFORM  # alias -- same transform, named for its Mahalanobis-fitting use


class _Subset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base_dataset[self.indices[i]]
        return self.transform(img), label


def load_cifar10(root):
    train_dataset = CIFAR10(root=root, train=True, download=True)
    test_dataset = CIFAR10(root=root, train=False, download=True)
    return train_dataset, test_dataset


def make_loader(base_dataset, indices, transform, batch_size=128, shuffle=False, drop_last=False, num_workers=2):
    subset = _Subset(base_dataset, indices, transform)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
