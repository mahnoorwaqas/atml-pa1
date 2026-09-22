"""
Common Evaluation and Alignment Diagnostic (Step 5). Loads all four trained
checkpoints and produces the required comparison table, domain-separability
scores, and per-class analysis vs. Source-only.

Run from the REPO ROOT, after training all four methods with train.py:
    python3 task2/evaluate_final.py --config task2/configs/base.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from shared.pacs import load_all_source_domains, load_target_domain, PACS_CLASSES
from shared.pacs_protocol import build_source_splits, make_eval_loader
from task2.models.backbone import build_resnet18
from task2.evaluation.metrics import evaluate_all_domains, build_comparison_table
from task2.evaluation.domain_separability import domain_separability_score
from task2.evaluation.class_analysis import per_class_accuracy, compare_per_class, confusion_for_class

CHECKPOINTS = {
    "source_only": "task2/results/checkpoints/source_only.pt",
    "dan": "task2/results/checkpoints/dan.pt",
    "dann": "task2/results/checkpoints/dann.pt",
    "cdan": "task2/results/checkpoints/cdan.pt",
}


def load_trained_model(checkpoint_path, n_classes, device):
    """
    Builds the plain classifier architecture (no domain_discriminator) and
    loads the checkpoint with strict=False: DANN/CDAN checkpoints also
    contain domain_discriminator.* weights that this eval-time architecture
    doesn't define, which is fine -- the discriminator is training-only
    scaffolding, irrelevant to classification/feature evaluation.
    """
    model = build_resnet18(n_classes=n_classes, weights=None)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    splits = build_source_splits(cfg["data"]["root"], seed=cfg["seed"])
    source_domains = load_all_source_domains(cfg["data"]["root"])
    source_val_loaders = {
        name: make_eval_loader(folder, splits[name]["val_indices"])
        for name, folder in source_domains.items()
    }
    target_folder = load_target_domain(cfg["data"]["root"])
    target_loader = make_eval_loader(target_folder)

    results_by_method = {}
    separability_by_method = {}
    per_class_by_method = {}

    for method_name, ckpt_path in tqdm(
        CHECKPOINTS.items(),
        desc="Evaluating methods",
    ):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: no checkpoint at {ckpt_path} -- skipping {method_name}. Train it first.")
            continue

        model = load_trained_model(ckpt_path, cfg["data"]["n_classes"], device)

        results_by_method[method_name] = evaluate_all_domains(
            model, source_val_loaders, target_loader, device
        )

        separability_by_method[method_name] = domain_separability_score(
            model, source_val_loaders, target_loader, device, seed=cfg["seed"]
        )

        accs, preds, labels = per_class_accuracy(
            model, target_loader, device, cfg["data"]["n_classes"]
        )

        per_class_by_method[method_name] = {
            "accs": accs,
            "preds": preds.tolist(),
            "labels": labels.tolist(),
        }

    if "source_only" not in results_by_method:
        print("No source_only checkpoint found -- can't build the comparison table (needs a baseline). Exiting.")
        return

    table = build_comparison_table(results_by_method)
    for method_name in table:
        table[method_name]["domain_separability"] = separability_by_method[method_name]

    os.makedirs("task2/results", exist_ok=True)
    with open("task2/results/comparison_table.json", "w") as f:
        json.dump(table, f, indent=2)

    baseline_accs = per_class_by_method["source_only"]["accs"]
    per_class_report = {}
    for method_name in ("dan", "dann", "cdan"):
        if method_name not in per_class_by_method:
            continue

        method_accs = per_class_by_method[method_name]["accs"]
        comparison = compare_per_class(
            baseline_accs,
            method_accs,
            PACS_CLASSES
        )

        preds_np = np.array(per_class_by_method[method_name]["preds"])
        labels_np = np.array(per_class_by_method[method_name]["labels"])

        confusions = {}
        for class_name, _ in (
            comparison["largest_improvement"]
            + comparison["largest_degradation"]
        ):
            class_idx = PACS_CLASSES.index(class_name)
            confusions[class_name] = confusion_for_class(
                preds_np,
                labels_np,
                class_idx,
                PACS_CLASSES
            )

        per_class_report[method_name] = {
            **comparison,
            "confusions": confusions
        }

    with open("task2/results/per_class_analysis.json", "w") as f:
        json.dump(per_class_report, f, indent=2)

    print(json.dumps(table, indent=2))
    print(json.dumps(per_class_report, indent=2))


if __name__ == "__main__":
    main()