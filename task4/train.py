"""
CLI entry point for Task 4. Dispatches to vanilla/gcsc/proser's training
functions based on config. PROSER requires Vanilla to be trained first
(it initializes from that checkpoint).

Run from the REPO ROOT:
    python3 task4/train.py --config task4/configs/vanilla.yaml
    python3 task4/train.py --config task4/configs/gcsc.yaml
    python3 task4/train.py --config task4/configs/proser.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from torchvision import transforms as tvt

from task4.data.cifar10 import load_cifar10, make_loader, build_train_transform, EVAL_TRANSFORM
from task4.data.make_splits import build_cifar10_split
from task4.methods import vanilla as vanilla_module
from task4.methods import gcsc as gcsc_module
from task4.methods import proser as proser_module


def _device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device()
    method_name = cfg["method"]["name"]

    split = build_cifar10_split(cfg["data"]["root"], seed=cfg["seed"])
    train_dataset, test_dataset = load_cifar10(cfg["data"]["root"])

    if method_name == "vanilla":
        train_transform = build_train_transform()
        model = vanilla_module.build_model(n_classes=cfg["data"]["n_classes"]).to(device)
        train_fn = vanilla_module.train_cifar_classifier

    elif method_name == "gcsc":
        randaugment = tvt.RandAugment(
            num_ops=cfg["method"].get("randaugment_num_ops", 2),
            magnitude=cfg["method"].get("randaugment_magnitude", 9),
        )
        train_transform = build_train_transform(extra_transform=randaugment)
        model = gcsc_module.build_model(n_classes=cfg["data"]["n_classes"]).to(device)
        train_fn = gcsc_module.train_cifar_classifier

    elif method_name == "proser":
        train_transform = build_train_transform()
        model = proser_module.build_model(
            n_classes=cfg["data"]["n_classes"],
            vanilla_checkpoint_path=cfg["vanilla_checkpoint_path"],
            device=device,
        ).to(device)
        train_fn = proser_module.train_proser

    else:
        raise ValueError(f"Unknown method '{method_name}'")

    train_loader = make_loader(
        train_dataset, split["train_indices"], train_transform,
        batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True,
    )
    val_loader = make_loader(
        train_dataset, split["val_indices"], EVAL_TRANSFORM,
        batch_size=cfg["training"]["batch_size"], shuffle=False,
    )

    torch.manual_seed(cfg["seed"])

    history, best_val_acc = train_fn(
        model, train_loader, val_loader, device, cfg, cfg["checkpoint_path"], seed=cfg["seed"],
    )

    log_path = cfg["checkpoint_path"].replace(".pt", "_training_log.json")
    with open(log_path, "w") as f:
        json.dump({"history": history, "best_val_accuracy": best_val_acc}, f, indent=2)

    print(f"Done. Best CIFAR-10 validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {cfg['checkpoint_path']}")


if __name__ == "__main__":
    main()
