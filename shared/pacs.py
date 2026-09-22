"""
PACS dataset loading + domain metadata.

Expects the standard on-disk layout used by most public PACS mirrors:
    <root>/<domain>/<class_name>/<image>.jpg
where <domain> in {photo, art_painting, cartoon, sketch} and there are 7
class folders per domain (dog, elephant, giraffe, guitar, horse, house,
person -- PACS's standard 7 classes).

PACS isn't bundled with torchvision, and its official distribution is a
gated Google Drive link with no stable direct-download URL, so this module
does NOT attempt to download it automatically -- place the extracted
dataset at the path in configs/base.yaml's data.root before running
anything that imports this file.
"""
import os

from torchvision.datasets import ImageFolder

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load_domain(root, domain, transform=None):
    """
    Returns a torchvision ImageFolder over <root>/<domain>. Raises a clear
    error if the folder is missing rather than a cryptic torchvision one.
    """
    domain_root = os.path.join(root, domain)
    if not os.path.isdir(domain_root):
        raise FileNotFoundError(
            f"Expected PACS domain folder at {domain_root} -- "
            f"see shared/pacs.py's docstring for the expected layout."
        )
    dataset = ImageFolder(domain_root, transform=transform)
    if sorted(dataset.classes) != sorted(PACS_CLASSES):
        raise ValueError(
            f"{domain_root} classes {sorted(dataset.classes)} don't match "
            f"expected PACS classes {sorted(PACS_CLASSES)}."
        )
    return dataset


def load_all_source_domains(root, transform=None):
    """Returns {domain_name: ImageFolder} for the three source domains."""
    return {d: load_domain(root, d, transform=transform) for d in SOURCE_DOMAINS}


def load_target_domain(root, transform=None):
    return load_domain(root, TARGET_DOMAIN, transform=transform)
