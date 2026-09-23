"""
CIFAR-100 test-set filtering to the fixed near/far unknown classes. The
grouping is FIXED by the assignment and must not be revised after seeing
results.

Each CIFAR-100 fine class has exactly 100 official test images, so 8 classes
per group gives exactly 800 images -- no sampling needed, just filtering by
class name.
"""
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100

from task4.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from torchvision import transforms as tvt

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

EVAL_TRANSFORM = tvt.Compose([tvt.ToTensor(), tvt.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


class _FilteredSubset(Dataset):
    """
    Returns (image, fine_label) where fine_label is the CIFAR-100 fine-class
    index -- NOT a CIFAR-10 label. Use it only to identify which unknown
    class an example came from (e.g. for failure analysis); never pass it to
    a loss against the 10-way classifier.
    """

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.classes = base_dataset.classes  # exposed for fine_label -> name lookup

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, fine_label = self.base_dataset[self.indices[i]]
        return self.transform(img), fine_label


def load_cifar100_unknowns(root):
    """
    Returns (near_dataset, far_dataset). CIFAR-100 TRAINING images are never
    touched here (test-set only), per the assignment.
    """
    test_dataset = CIFAR100(root=root, train=False, download=True)
    class_names = test_dataset.classes

    near_ids = {class_names.index(c) for c in NEAR_CLASSES}
    far_ids = {class_names.index(c) for c in FAR_CLASSES}

    near_indices = [i for i, label in enumerate(test_dataset.targets) if label in near_ids]
    far_indices = [i for i, label in enumerate(test_dataset.targets) if label in far_ids]

    assert len(near_indices) == 800, f"expected 800 near-unknown images, got {len(near_indices)}"
    assert len(far_indices) == 800, f"expected 800 far-unknown images, got {len(far_indices)}"

    near_dataset = _FilteredSubset(test_dataset, near_indices, EVAL_TRANSFORM)
    far_dataset = _FilteredSubset(test_dataset, far_indices, EVAL_TRANSFORM)
    return near_dataset, far_dataset


def make_loader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
