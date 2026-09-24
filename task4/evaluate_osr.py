"""
Common Evaluation and Failure Analysis (Step 6). Loads cached outputs
(extract_outputs.py must be run first for vanilla/gcsc/proser) and produces:
  - Table 1: MSP/MLS/Energy/Mahalanobis on the frozen Vanilla model
    (near/far/all AUROC + validation-calibrated rejection).
  - Table 2: Vanilla/GCSC/PROSER compared with MLS as the common score, plus
    a PROSER placeholder-score row.
  - A compact score-distribution figure (MSP/MLS/Mahalanobis).
  - Failure analysis: 3 incorrectly-accepted near + 3 far examples, at the
    Vanilla MLS threshold.

Run from the REPO ROOT, after extract_outputs.py for each trained method:
    python3 task4/evaluate_osr.py --config task4/configs/vanilla.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from task4.data.hf_cifar import CIFAR10HF as CIFAR10, CIFAR100HF as CIFAR100

from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import fit_mahalanobis_stats, mahalanobis_score
from task4.evaluation.metrics import auroc_known_vs_unknown, closed_set_accuracy
from task4.evaluation.thresholds import calibrate_threshold, acceptance_and_rejection_rates
from task4.evaluation.failure_analysis import incorrectly_accepted
from task4.methods.proser import proser_detection_score

CACHE_DIR = "task4/cache"
RESULTS_DIR = "task4/results"


def _load_cached(method_name):
    path = os.path.join(CACHE_DIR, f"{method_name}_outputs.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run extract_outputs.py for '{method_name}' first.")
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cifar10_classes = CIFAR10(root=cfg["data"]["root"], train=False, download=True).classes
    cifar100_classes = CIFAR100(root=cfg["data"]["root"], train=False, download=True).classes

    # ---------- Table 1: MSP/MLS/Energy/Mahalanobis on the frozen Vanilla model ----------
    vanilla = _load_cached("vanilla")

    train_feats = vanilla["train"]["features"].numpy()
    train_labels = vanilla["train"]["labels"].numpy()
    class_means, inv_diag_cov = fit_mahalanobis_stats(train_feats, train_labels, n_classes=cfg["data"]["n_classes"])

    def _all_scores(split):
        logits = vanilla[split]["logits"].numpy()
        feats = vanilla[split]["features"].numpy()
        return {
            "msp": msp_score(logits),
            "mls": mls_score(logits),
            "energy": energy_score(logits),
            "mahalanobis": mahalanobis_score(feats, class_means, inv_diag_cov),
        }

    val_scores = _all_scores("val")
    test_scores = _all_scores("test")
    near_scores = _all_scores("near")
    far_scores = _all_scores("far")

    table1 = {}
    for score_name in ["msp", "mls", "energy", "mahalanobis"]:
        threshold = calibrate_threshold(val_scores[score_name], percentile=95.0)
        rates = acceptance_and_rejection_rates(
            test_scores[score_name], near_scores[score_name], far_scores[score_name], threshold
        )
        table1[score_name] = {
            "auroc_near": auroc_known_vs_unknown(test_scores[score_name], near_scores[score_name]),
            "auroc_far": auroc_known_vs_unknown(test_scores[score_name], far_scores[score_name]),
            "auroc_all": auroc_known_vs_unknown(
                test_scores[score_name], np.concatenate([near_scores[score_name], far_scores[score_name]])
            ),
            **rates,
        }

    with open(os.path.join(RESULTS_DIR, "table1_vanilla_scores.json"), "w") as f:
        json.dump(table1, f, indent=2)
    print(json.dumps(table1, indent=2))

    # ---------- score-distribution figure (MSP, MLS, Mahalanobis) ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score_name in zip(axes, ["msp", "mls", "mahalanobis"]):
        ax.hist(test_scores[score_name], bins=40, alpha=0.5, label="known (test)", density=True)
        ax.hist(near_scores[score_name], bins=40, alpha=0.5, label="near unknown", density=True)
        ax.hist(far_scores[score_name], bins=40, alpha=0.5, label="far unknown", density=True)
        ax.set_title(score_name.upper())
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "score_distributions.png"), dpi=150)
    plt.close(fig)

    # ---------- Table 2: Vanilla / GCSC / PROSER with MLS, + PROSER placeholder row ----------
    table2 = {}
    for method_name in ["vanilla", "gcsc", "proser"]:
        cache_path = os.path.join(CACHE_DIR, f"{method_name}_outputs.pt")
        if not os.path.exists(cache_path):
            print(f"WARNING: no cached outputs for '{method_name}' -- run extract_outputs.py first. Skipping.")
            continue
        cached = torch.load(cache_path)

        csa = closed_set_accuracy(cached["test"]["logits"].numpy(), cached["test"]["labels"].numpy())

        mls_val = mls_score(cached["val"]["logits"].numpy())
        mls_test = mls_score(cached["test"]["logits"].numpy())
        mls_near = mls_score(cached["near"]["logits"].numpy())
        mls_far = mls_score(cached["far"]["logits"].numpy())
        threshold = calibrate_threshold(mls_val, percentile=95.0)
        rates = acceptance_and_rejection_rates(mls_test, mls_near, mls_far, threshold)

        table2[method_name] = {
            "closed_set_accuracy": csa,
            "auroc_near_mls": auroc_known_vs_unknown(mls_test, mls_near),
            "auroc_far_mls": auroc_known_vs_unknown(mls_test, mls_far),
            "auroc_all_mls": auroc_known_vs_unknown(mls_test, np.concatenate([mls_near, mls_far])),
            **rates,
        }

        if method_name == "proser" and "dummy_logits" in cached["val"]:
            ph_val = proser_detection_score(cached["val"]["logits"].numpy(), cached["val"]["dummy_logits"].numpy())
            ph_test = proser_detection_score(cached["test"]["logits"].numpy(), cached["test"]["dummy_logits"].numpy())
            ph_near = proser_detection_score(cached["near"]["logits"].numpy(), cached["near"]["dummy_logits"].numpy())
            ph_far = proser_detection_score(cached["far"]["logits"].numpy(), cached["far"]["dummy_logits"].numpy())
            ph_threshold = calibrate_threshold(ph_val, percentile=95.0)
            ph_rates = acceptance_and_rejection_rates(ph_test, ph_near, ph_far, ph_threshold)

            table2["proser_placeholder_score"] = {
                "closed_set_accuracy": csa,  # same CSA as the MLS row -- CSA depends only on known logits
                "auroc_near": auroc_known_vs_unknown(ph_test, ph_near),
                "auroc_far": auroc_known_vs_unknown(ph_test, ph_far),
                "auroc_all": auroc_known_vs_unknown(ph_test, np.concatenate([ph_near, ph_far])),
                **ph_rates,
            }

    with open(os.path.join(RESULTS_DIR, "table2_method_comparison.json"), "w") as f:
        json.dump(table2, f, indent=2)
    print(json.dumps(table2, indent=2))

    # ---------- failure analysis: Vanilla MLS threshold, 3 near + 3 far ----------
    vanilla_mls_val = mls_score(vanilla["val"]["logits"].numpy())
    vanilla_mls_threshold = calibrate_threshold(vanilla_mls_val, percentile=95.0)

    near_preds = vanilla["near"]["logits"].numpy().argmax(axis=1)
    far_preds = vanilla["far"]["logits"].numpy().argmax(axis=1)
    near_fine_labels = vanilla["near"]["labels"].numpy()
    far_fine_labels = vanilla["far"]["labels"].numpy()

    near_failures = incorrectly_accepted(
        near_scores["mls"], near_preds, near_fine_labels, vanilla_mls_threshold,
        cifar10_classes, cifar100_classes, n_examples=3, seed=cfg["seed"],
    )
    far_failures = incorrectly_accepted(
        far_scores["mls"], far_preds, far_fine_labels, vanilla_mls_threshold,
        cifar10_classes, cifar100_classes, n_examples=3, seed=cfg["seed"],
    )

    with open(os.path.join(RESULTS_DIR, "failure_analysis.json"), "w") as f:
        json.dump({"near": near_failures, "far": far_failures}, f, indent=2)
    print(json.dumps({"near": near_failures, "far": far_failures}, indent=2))


if __name__ == "__main__":
    main()
