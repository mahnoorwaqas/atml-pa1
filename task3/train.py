"""
CLI entry point for Task 3. ERM is NOT trained here -- it's Task 2's
Source-only checkpoint, reused unchanged (running erm.yaml just verifies
the checkpoint exists). DAN-DG and SAM are trained fresh, using ONLY the
three source domains -- no Sketch image is ever loaded by this script.

Run from the REPO ROOT:
    python3 task3/train.py --config task3/configs/erm.yaml      # verifies Task 2's checkpoint exists
    python3 task3/train.py --config task3/configs/dan_dg.yaml
    python3 task3/train.py --config task3/configs/sam.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from shared.pacs import load_all_source_domains
from shared.pacs_protocol import (
    build_source_splits, make_domain_balanced_loaders, make_eval_loader,
    default_steps_per_epoch,
)
from shared.train_utils import train_loop
from task3.methods import dan_dg as dan_dg_module
from task3.methods import sam as sam_module


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" not in cfg:
        return cfg
    base_path = os.path.join(os.path.dirname(path), cfg["defaults"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    merged = {**base_cfg, **{k: v for k, v in cfg.items() if k != "defaults"}}
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method_name == "erm":
        if not os.path.exists(cfg["checkpoint_path"]):
            raise FileNotFoundError(
                f"{cfg['checkpoint_path']} not found -- train Task 2's "
                "source_only method first: python3 task2/train.py --config "
                "task2/configs/source_only.yaml"
            )
        print(f"ERM baseline: reusing Task 2's checkpoint at {cfg['checkpoint_path']} unchanged. Nothing to train.")
        return

    splits = build_source_splits(cfg["data"]["root"], seed=cfg["seed"])
    steps_per_epoch = cfg["training"]["steps_per_epoch"] or default_steps_per_epoch(
        splits, per_source=cfg["data"]["per_source_batch"]
    )

    # include_target=False, ALWAYS -- Task 3 must never load a Sketch image
    # during training, per the assignment.
    batch_iterator = make_domain_balanced_loaders(
        cfg["data"]["root"], splits,
        per_source=cfg["data"]["per_source_batch"],
        per_target=0, include_target=False, seed=cfg["seed"],
    )

    source_domains = load_all_source_domains(cfg["data"]["root"])
    source_val_loaders = {
        name: make_eval_loader(folder, splits[name]["val_indices"])
        for name, folder in source_domains.items()
    }

    torch.manual_seed(cfg["seed"])

    if method_name == "dan_dg":
        model = dan_dg_module.build_model(cfg).to(device)
        compute_loss_fn = dan_dg_module.make_compute_loss(cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
        )
        history, best_val_f1 = train_loop(
            model, batch_iterator, optimizer, compute_loss_fn,
            source_val_loaders, cfg, cfg["checkpoint_path"], device,
            max_steps_per_epoch=steps_per_epoch,
        )

    elif method_name == "sam":
        model = sam_module.build_model(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
        )
        history, best_val_f1 = sam_module.train_sam(
            model, batch_iterator, optimizer, cfg["method"]["rho"],
            source_val_loaders, cfg, cfg["checkpoint_path"], device,
            max_steps_per_epoch=steps_per_epoch,
        )

    else:
        raise ValueError(f"Unknown method '{method_name}'")

    log_path = cfg["checkpoint_path"].replace(".pt", "_training_log.json")
    with open(log_path, "w") as f:
        json.dump({"history": history, "best_val_macro_f1": best_val_f1}, f, indent=2)

    print(f"Done. Best mean source-validation macro-F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to {cfg['checkpoint_path']}")


if __name__ == "__main__":
    main()
