"""
Step 6: Controlled Design Study. Runs ONE of the two bounded sweeps (choose
by which config you pass -- dan.yaml sweeps lambda_mmd, dann.yaml sweeps
max_grl_strength, per each config's controlled_study section), training a
fresh model per value. Target results here are FOR ANALYSIS ONLY -- the
assignment explicitly says these may NOT be used to revise the main
comparison's settings (which always stays at lambda_mmd=1 / max_grl_strength=1
in evaluate_final.py, regardless of what this sweep finds).

Run from the REPO ROOT (choose ONE):
    python3 task2/run_controlled_study.py --config task2/configs/dan.yaml
    python3 task2/run_controlled_study.py --config task2/configs/dann.yaml
"""
import argparse
import copy
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from shared.pacs import load_all_source_domains, load_target_domain
from shared.pacs_protocol import (
    build_source_splits, make_domain_balanced_loaders, make_eval_loader,
    default_steps_per_epoch,
)
from shared.train_utils import train_loop
from task2.evaluation.metrics import evaluate_all_domains
from task2.evaluation.domain_separability import domain_separability_score
from task2.train import load_config, METHOD_MODULES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_name = cfg["method"]["name"]
    sweep_param = cfg["controlled_study"]["sweep_param"]
    sweep_values = cfg["controlled_study"]["values"]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    module = importlib.import_module(METHOD_MODULES[method_name])

    splits = build_source_splits(cfg["data"]["root"], seed=cfg["seed"])
    steps_per_epoch = cfg["training"]["steps_per_epoch"] or default_steps_per_epoch(
        splits, per_source=cfg["data"]["per_source_batch"]
    )
    source_domains = load_all_source_domains(cfg["data"]["root"])
    source_val_loaders = {
        name: make_eval_loader(folder, splits[name]["val_indices"])
        for name, folder in source_domains.items()
    }
    target_folder = load_target_domain(cfg["data"]["root"])
    target_loader = make_eval_loader(target_folder)

    sweep_results = {}

    for value in sweep_values:
        print(f"--- {sweep_param} = {value} ---")
        sweep_cfg = copy.deepcopy(cfg)
        sweep_cfg["method"][sweep_param] = value

        batch_iterator = make_domain_balanced_loaders(
            cfg["data"]["root"], splits,
            per_source=cfg["data"]["per_source_batch"],
            per_target=cfg["data"]["per_target_batch"],
            include_target=True,
            seed=cfg["seed"],
        )

        model = module.build_model(sweep_cfg).to(device)
        compute_loss_fn = module.make_compute_loss(sweep_cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
        )
        torch.manual_seed(cfg["seed"])

        ckpt_path = f"task2/results/checkpoints/{method_name}_{sweep_param}_{value}.pt"
        _, best_val_f1 = train_loop(
            model, batch_iterator, optimizer, compute_loss_fn,
            source_val_loaders, cfg, ckpt_path, device,
            max_steps_per_epoch=steps_per_epoch,
        )

        eval_result = evaluate_all_domains(model, source_val_loaders, target_loader, device)
        separability = domain_separability_score(model, source_val_loaders, target_loader, device, seed=cfg["seed"])

        sweep_results[str(value)] = {
            **eval_result,
            "domain_separability": separability,
            "best_val_macro_f1": best_val_f1,
        }
        print(sweep_results[str(value)])

    out_path = f"task2/results/controlled_study_{method_name}_{sweep_param}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
