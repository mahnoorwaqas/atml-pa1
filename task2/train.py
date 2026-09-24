"""
CLI entry point: trains one method (source_only / dan / dann / cdan) per its
config file. Loads base.yaml first, then the method-specific yaml overlays
it (via a "defaults: base.yaml" key).

Run from the REPO ROOT:
    python3 task2/train.py --config task2/configs/source_only.yaml
    python3 task2/train.py --config task2/configs/dan.yaml
    python3 task2/train.py --config task2/configs/dann.yaml --backbone-lr-mult 0.1
    python3 task2/train.py --config task2/configs/cdan.yaml --backbone-lr-mult 0.1

Train source_only FIRST -- its checkpoint isn't required by the others at
training time, but Task 3 reuses it unchanged as the ERM baseline, and
evaluate_final.py needs it present to compute accuracy deltas.

Learning-rate policy: the classifier head (model.fc) and, for DANN/CDAN, the
domain discriminator use the base lr. The pretrained backbone uses
lr * backbone_lr_mult. The multiplier defaults to 1.0 (identical to a single
learning rate), so source_only and DAN are unchanged unless you pass it.
It can be set with --backbone-lr-mult or via training.backbone_lr_mult in a
yaml config; the CLI flag wins if both are given.
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from shared.pacs_protocol import (
    build_source_splits, make_domain_balanced_loaders, make_eval_loader,
    default_steps_per_epoch,
)
from shared.pacs import load_all_source_domains
from shared.train_utils import train_loop

METHOD_MODULES = {
    "source_only": "task2.methods.source_only",
    "dan": "task2.methods.dan",
    "dann": "task2.methods.dann",
    "cdan": "task2.methods.cdan",
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" not in cfg:
        return cfg

    base_path = os.path.join(os.path.dirname(path), cfg["defaults"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    merged = {**base_cfg, **{k: v for k, v in cfg.items() if k != "defaults"}}
    # Shallow-merge nested dicts rather than letting the method yaml fully
    # overwrite e.g. "training" if it only overrides one field of it.
    for key in ("data", "model", "training"):
        if key in base_cfg:
            merged[key] = {**base_cfg[key], **cfg.get(key, {})}
    return merged


def build_optimizer(model, cfg, backbone_lr_mult):
    """
    AdamW with two param groups:
      - head group: model.fc (+ model.domain_discriminator if present), lr
      - backbone group: everything else (conv layers + BN gamma/beta),
        lr * backbone_lr_mult
    """
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"]["weight_decay"]

    head_params = list(model.fc.parameters())
    if hasattr(model, "domain_discriminator"):
        head_params += list(model.domain_discriminator.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    print(
        f"Optimizer: backbone lr = {lr * backbone_lr_mult:g} "
        f"({len(backbone_params)} tensors), head/discriminator lr = {lr:g} "
        f"({len(head_params)} tensors)"
    )
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backbone-lr-mult", type=float, default=None,
        help="Multiplier on the base lr for the pretrained backbone "
             "(default: training.backbone_lr_mult in the config, else 1.0).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_name = cfg["method"]["name"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    backbone_lr_mult = (
        args.backbone_lr_mult
        if args.backbone_lr_mult is not None
        else cfg["training"].get("backbone_lr_mult", 1.0)
    )

    module = importlib.import_module(METHOD_MODULES[method_name])

    splits = build_source_splits(cfg["data"]["root"], seed=cfg["seed"])
    steps_per_epoch = cfg["training"]["steps_per_epoch"] or default_steps_per_epoch(
        splits, per_source=cfg["data"]["per_source_batch"]
    )

    include_target = method_name != "source_only"  # Source-only never touches target data
    batch_iterator = make_domain_balanced_loaders(
        cfg["data"]["root"], splits,
        per_source=cfg["data"]["per_source_batch"],
        per_target=cfg["data"]["per_target_batch"],
        include_target=include_target,
        seed=cfg["seed"],
    )

    source_domains = load_all_source_domains(cfg["data"]["root"])
    source_val_loaders = {
        name: make_eval_loader(folder, splits[name]["val_indices"])
        for name, folder in source_domains.items()
    }

    model = module.build_model(cfg).to(device)
    compute_loss_fn = module.make_compute_loss(cfg)

    optimizer = build_optimizer(model, cfg, backbone_lr_mult)

    torch.manual_seed(cfg["seed"])

    history, best_val_f1 = train_loop(
        model, batch_iterator, optimizer, compute_loss_fn,
        source_val_loaders, cfg, cfg["checkpoint_path"], device,
        max_steps_per_epoch=steps_per_epoch,
    )

    log_path = cfg["checkpoint_path"].replace(".pt", "_training_log.json")
    with open(log_path, "w") as f:
        json.dump({"history": history, "best_val_macro_f1": best_val_f1}, f, indent=2)

    print(f"Done. Best mean source-validation macro-F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to {cfg['checkpoint_path']}")


if __name__ == "__main__":
    main()