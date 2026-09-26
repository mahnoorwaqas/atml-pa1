"""
Plots the Task 1 translation curves (accuracy and consistency vs.
displacement) from translation_metrics.json.

Usage:
    python plot_translation_curve.py --input translation_metrics.json --outdir results
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


DISPLACEMENTS = [0, 8, 16, 32]

MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
    "clip_vit_b_32": "CLIP ViT-B/32",
}

# Fixed colors so this plot's palette matches any other Task 1 figures
# that use the same models (e.g. if you re-use these for the t-SNE panels).
MODEL_COLORS = {
    "resnet50": "#1f77b4",
    "vit_b_16": "#ff7f0e",
    "clip_vit_b_32": "#2ca02c",
}


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def extract_series(metrics, model_key, field):
    """field: 'accuracy' or 'consistency'."""
    return [metrics[model_key][str(d)][field] for d in DISPLACEMENTS]


def plot_curve(metrics, field, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for model_key, label in MODEL_LABELS.items():
        if model_key not in metrics:
            continue
        y = extract_series(metrics, model_key, field)
        ax.plot(
            DISPLACEMENTS, y,
            marker="o", label=label,
            color=MODEL_COLORS.get(model_key),
        )
    ax.set_xlabel("Displacement (pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(DISPLACEMENTS)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_combined(metrics, save_path):
    """
    Single figure, two panels side by side: accuracy (left) and
    consistency (right) vs. displacement. This is the version to use
    in the report if you want one compact Figure rather than two.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)

    for model_key, label in MODEL_LABELS.items():
        if model_key not in metrics:
            continue
        color = MODEL_COLORS.get(model_key)
        acc = extract_series(metrics, model_key, "accuracy")
        cons = extract_series(metrics, model_key, "consistency")
        axes[0].plot(DISPLACEMENTS, acc, marker="o", label=label, color=color)
        axes[1].plot(DISPLACEMENTS, cons, marker="o", label=label, color=color)

    axes[0].set_title("Accuracy vs. displacement")
    axes[0].set_xlabel("Displacement (pixels)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(DISPLACEMENTS)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Consistency vs. displacement")
    axes[1].set_xlabel("Displacement (pixels)")
    axes[1].set_ylabel("Prediction consistency")
    axes[1].set_xticks(DISPLACEMENTS)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="translation_metrics.json")
    parser.add_argument("--outdir", default="results")
    parser.add_argument(
        "--mode", choices=["separate", "combined"], default="combined",
        help="'combined': one 2-panel figure (accuracy | consistency), "
             "recommended for the report given limited space. "
             "'separate': two standalone figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    metrics = load_metrics(args.input)

    if args.mode == "combined":
        plot_combined(metrics, os.path.join(args.outdir, "translation_curve.png"))
    else:
        plot_curve(
            metrics, "accuracy", "Accuracy",
            "Accuracy vs. displacement",
            os.path.join(args.outdir, "translation_accuracy_curve.png"),
        )
        plot_curve(
            metrics, "consistency", "Prediction consistency",
            "Consistency vs. displacement",
            os.path.join(args.outdir, "translation_consistency_curve.png"),
        )


if __name__ == "__main__":
    main()