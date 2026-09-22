"""
CLI entry point: trains one method (source_only / dan / dann / cdan) per its
config file. Loads base.yaml first, then the method-specific yaml overlays
it (via a "defaults: base.yaml" key).

Run from the REPO ROOT:
    python3 task2/train.py --config task2/configs/source_only.yaml
    python3 task2/train.py --config task2/configs/dan.yaml
    python3 task2/train.py --config task2/configs/dann.yaml
    python3 task2/train.py --config task2/configs/cdan.yaml

Train source_only FIRST -- its checkpoint isn't required by the others at
training time, but Task 3 reuses it unchanged as the ERM baseline, and
evaluate_final.py needs it present to compute accuracy deltas.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_name = cfg["method"]["name"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

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
