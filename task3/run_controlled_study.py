"""
Step 5: Controlled Design Study. Sweeps EITHER dan_dg.yaml's lambda_dg OR
sam.yaml's rho (the assignment says choose ONE). The main comparison stays
at lambda_dg=1 / rho=0.05 regardless of what this sweep finds.

This script deliberately reports ONLY source-side diagnostics (mean/worst-
source, source-domain separability, sharpness proxy) for each swept value --
it never loads Sketch, keeping the sweep itself uncontaminated by target
information. If you want the "analysis only" Sketch numbers the assignment
allows for interpreting the controlled study, evaluate the saved
per-value checkpoints with evaluate_sketch.py's logic SEPARATELY, after
every value's checkpoint already exists.

Run from the REPO ROOT (choose ONE):
    python3 task3/run_controlled_study.py --config task3/configs/dan_dg.yaml
    python3 task3/run_controlled_study.py --config task3/configs/sam.yaml
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm

from shared.pacs import load_all_source_domains
from shared.pacs_protocol import (
    build_source_splits, make_domain_balanced_loaders, make_eval_loader,
    default_steps_per_epoch,
)
from shared.train_utils import train_loop
from task3.evaluation.source_domain_separability import source_domain_separability_score
from task3.evaluation.sharpness import build_fixed_sharpness_batch, sharpness_proxy
from task3.selection.source_validation import per_domain_and_worst
from task3.train import load_config
from task3.methods import dan_dg as dan_dg_module
from task3.methods import sam as sam_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_name = cfg["method"]["name"]
    sweep_param = cfg["controlled_study"]["sweep_param"]
    sweep_values = cfg["controlled_study"]["values"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = build_source_splits(cfg["data"]["root"], seed=cfg["seed"])
    steps_per_epoch = cfg["training"]["steps_per_epoch"] or default_steps_per_epoch(
        splits, per_source=cfg["data"]["per_source_batch"]
    )
    source_domains = load_all_source_domains(cfg["data"]["root"])
    source_val_loaders = {
        name: make_eval_loader(folder, splits[name]["val_indices"])
        for name, folder in source_domains.items()
    }
    fixed_images, fixed_labels = build_fixed_sharpness_batch(
        source_val_loaders, n_per_domain=32, seed=cfg["seed"]
    )

    sweep_results = {}

    for value in tqdm(sweep_values, desc=f"{sweep_param} sweep"):
        print(f"--- {sweep_param} = {value} ---")
        sweep_cfg = copy.deepcopy(cfg)
        sweep_cfg["method"][sweep_param] = value

        batch_iterator = make_domain_balanced_loaders(
            cfg["data"]["root"], splits,
            per_source=cfg["data"]["per_source_batch"],
            per_target=0, include_target=False, seed=cfg["seed"],
        )
        torch.manual_seed(cfg["seed"])
        ckpt_path = f"task3/results/checkpoints/{method_name}_{sweep_param}_{value}.pt"

        if method_name == "dan_dg":
            model = dan_dg_module.build_model(sweep_cfg).to(device)
            compute_loss_fn = dan_dg_module.make_compute_loss(sweep_cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
            )
            _, best_val_f1 = train_loop(
                model, batch_iterator, optimizer, compute_loss_fn,
                source_val_loaders, cfg, ckpt_path, device,
                max_steps_per_epoch=steps_per_epoch,
            )
        elif method_name == "sam":
            model = sam_module.build_model(sweep_cfg).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
            )
            _, best_val_f1 = sam_module.train_sam(
                model, batch_iterator, optimizer, sweep_cfg["method"]["rho"],
                source_val_loaders, cfg, ckpt_path, device,
                max_steps_per_epoch=steps_per_epoch,
            )
        else:
            raise ValueError(f"Controlled study not defined for method '{method_name}'")

        per_domain, worst_domain = per_domain_and_worst(model, source_val_loaders, device)
        mean_acc = sum(v["accuracy"] for v in per_domain.values()) / len(per_domain)
        mean_f1 = sum(v["macro_f1"] for v in per_domain.values()) / len(per_domain)
        separability = source_domain_separability_score(model, source_val_loaders, device, seed=cfg["seed"])
        sharpness = sharpness_proxy(model, fixed_images, fixed_labels, device, rho=0.05)

        sweep_results[str(value)] = {
            "per_source_domain": per_domain,
            "mean_source_accuracy": mean_acc,
            "mean_source_macro_f1": mean_f1,
            "worst_source_domain": worst_domain,
            "worst_source_macro_f1": per_domain[worst_domain]["macro_f1"],
            "source_domain_separability": separability,
            "sharpness_proxy": sharpness,
            "best_val_macro_f1": best_val_f1,
        }
        print(sweep_results[str(value)])

    out_path = f"task3/results/controlled_study_{method_name}_{sweep_param}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()