"""
Drop-in replacement for torchvision.datasets.CIFAR10 / CIFAR100 that reads
from the Hugging Face cache, so torchvision's MD5 check on its pickled batch
files never runs.

Same interface as the torchvision classes:
    .data, .targets, .classes, .class_to_idx, len(ds), ds[i] -> (img, target)

Requires: pip install datasets
Set HF_HUB_OFFLINE=1 to skip network checks once the data is cached.
If you downloaded under a different repo id, pass hf_id="..." explicitly.
"""
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class _HFCifar(VisionDataset):
    hf_id = None
    label_key = None

    def __init__(self, root=None, train=True, transform=None,
                 target_transform=None, download=False,
                 hf_id=None, cache_dir=None):
        # `root` and `download` are accepted for signature compatibility with
        # torchvision and are otherwise ignored.
        super().__init__(root, transform=transform, target_transform=target_transform)
        from datasets import load_dataset

        ds = load_dataset(
            hf_id or self.hf_id,
            split="train" if train else "test",
            cache_dir=cache_dir,
        )
        img_key = "img" if "img" in ds.column_names else "image"

        self.train = train
        self.data = np.stack(
            [np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds[img_key]]
        )
        self.targets = [int(t) for t in ds[self.label_key]]
        self.classes = list(ds.features[self.label_key].names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR10HF(_HFCifar):
    hf_id = "uoft-cs/cifar10"
    label_key = "label"


class CIFAR100HF(_HFCifar):
    hf_id = "uoft-cs/cifar100"
    label_key = "fine_label"
