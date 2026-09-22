"""
Generates cue-conflict images (shape/content = A, texture/style = B, and the
reverse direction) using AdaIN (naoto0804/pytorch-AdaIN, vendored in
task1/models/adain_net.py — see that file's docstring for attribution).

Before evaluating any model on the outputs, this enforces:
  - >= 5 unordered class pairs (cfg["cue_conflict"]["n_class_pairs"])
  - a fixed style-strength setting (cfg["cue_conflict"]["style_strength"])
  - a VISUAL rejection rule (passes_rejection_rule), defined BEFORE looking
    at any model prediction, using only image statistics
  - >= 200 valid (accepted) conflicts, balanced across pairs/directions
  - accepted/rejected counts recorded

Content and style images are drawn from the Task 1 dataset's TRAINING split
(distinct from the 500-image test subset used for clean/color/translation/
patch-shuffle evaluation), so cue-conflict generation doesn't touch the same
raw images used elsewhere.

Prerequisite: run download_adain_weights.py first.

Usage:
    python task1/data/make_cue_conflicts.py --config task1/configs/config.yaml
"""
import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as tvt
from torchvision.utils import save_image


from task1.data.make_subset import load_dataset, _get_labels
from task1.models import adain_net


def load_adain_model(vgg_weights_path, decoder_weights_path, device="cpu"):
    """Thin wrapper so this file's public interface matches the original
    skeleton's function name; delegates to adain_net.load_adain_model."""
    return adain_net.load_adain_model(vgg_weights_path, decoder_weights_path, device=device)


def sample_class_pairs(class_names, n_pairs, seed):
    """Select n_pairs unordered class pairs (as index tuples), fixed seed."""
    rng = random.Random(seed)
    all_pairs = list(itertools.combinations(range(len(class_names)), 2))
    rng.shuffle(all_pairs)
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} class pairs but only {len(all_pairs)} "
            f"unordered pairs exist for {len(class_names)} classes."
        )
    return all_pairs[:n_pairs]


def stylize(content_tensor, style_tensor, adain_model, alpha):
    """
    content_tensor, style_tensor: (1, 3, H, W) float tensors in [0, 1], same
    device as adain_model's weights.
    adain_model: (vgg_encoder, decoder_net) tuple from load_adain_model().
    Returns stylized (1, 3, H, W) tensor, clamped to [0, 1] (the decoder's raw
    output isn't guaranteed to stay in-range).
    """
    vgg_encoder, decoder_net = adain_model
    with torch.no_grad():
        output = adain_net.style_transfer(vgg_encoder, decoder_net, content_tensor, style_tensor, alpha=alpha)
    return output.clamp(0, 1)


def passes_rejection_rule(content_image, stylized_image, edge_corr_threshold=0.15, min_std_threshold=0.02):
    """
    Concrete, checkable rule, defined BEFORE looking at any model prediction
    (uses only image statistics, never model output):
      1. Reject if the stylized image is near-degenerate — pixel std too low
         (stylization collapsed to a near-uniform color/texture).
      2. Reject if the stylized image's edge map has too little correlation
         with the content image's edge map — indicates the object's shape
         silhouette was destroyed rather than merely re-textured.
    content_image, stylized_image: (3, H, W) arrays/tensors in [0, 1].
    Returns True if the image should be KEPT.
    """
    def _to_np(img):
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        return np.asarray(img)

    content = _to_np(content_image)
    stylized = _to_np(stylized_image)

    if stylized.std() < min_std_threshold:
        return False

    def _edge_map(img):
        gray = img.mean(axis=0)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        return np.sqrt(gx ** 2 + gy ** 2)

    content_edges = _edge_map(content).flatten()
    stylized_edges = _edge_map(stylized).flatten()

    if content_edges.std() == 0 or stylized_edges.std() == 0:
        return False

    correlation = np.corrcoef(content_edges, stylized_edges)[0, 1]
    if np.isnan(correlation) or correlation < edge_corr_threshold:
        return False

    return True


def _load_image_as_tensor(dataset, idx, size=224):
    img, _ = dataset[idx]
    img = img.convert("RGB")
    tf = tvt.Compose([tvt.Resize((size, size)), tvt.ToTensor()])
    return tf(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    cc_cfg = cfg["cue_conflict"]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _, class_names = load_dataset(cfg)
    labels = _get_labels(train_dataset)
    images_by_class = {c: np.where(labels == c)[0].tolist() for c in range(len(class_names))}

    class_pairs = sample_class_pairs(class_names, cc_cfg["n_class_pairs"], seed)

    weights_dir = os.path.join("models", "adain_weights")
    adain_model = load_adain_model(
        os.path.join(weights_dir, "vgg_normalised.pth"),
        os.path.join(weights_dir, "decoder.pth"),
        device=device,
    )

    combos = []
    for (a, b) in class_pairs:
        combos.append((a, b, "content_a_style_b"))
        combos.append((b, a, "content_b_style_a"))

    target_per_combo = -(-cc_cfg["min_valid_conflicts"] // len(combos))
    max_attempts_per_combo = target_per_combo * 5

    rng = random.Random(seed)
    out_dir = os.path.join("results", "cue_conflicts")
    os.makedirs(out_dir, exist_ok=True)

    accepted_meta = []
    rejected_count = 0

    for content_class, style_class, direction in combos:
        n_accepted = 0
        attempts = 0

        while n_accepted < target_per_combo and attempts < max_attempts_per_combo:
            attempts += 1
            content_idx = rng.choice(images_by_class[content_class])
            style_idx = rng.choice(images_by_class[style_class])

            content_tensor = _load_image_as_tensor(train_dataset, content_idx).to(device)
            style_tensor = _load_image_as_tensor(train_dataset, style_idx).to(device)

            stylized = stylize(
                content_tensor,
                style_tensor,
                adain_model,
                cc_cfg["style_strength"],
            )

            if passes_rejection_rule(content_tensor[0], stylized[0]):
                fname = f"{content_class}_{style_class}_{direction}_{n_accepted:03d}.png"
                save_path = os.path.join(out_dir, fname)
                save_image(stylized[0].cpu(), save_path)

                accepted_meta.append({
                    "path": save_path,
                    "content_class": content_class,
                    "style_class": style_class,
                    "direction": direction,
                    "content_idx": content_idx,
                    "style_idx": style_idx,
                })
                n_accepted += 1
            else:
                rejected_count += 1

        if n_accepted < target_per_combo:
            print(
                f"WARNING: only {n_accepted}/{target_per_combo} accepted for "
                f"combo (content={content_class}, style={style_class}, {direction}) "
                f"after {attempts} attempts."
            )

    n_total = len(accepted_meta)
    if n_total < cc_cfg["min_valid_conflicts"]:
        print(
            f"WARNING: total accepted conflicts ({n_total}) is below the "
            f"required minimum ({cc_cfg['min_valid_conflicts']})."
        )

    summary = {
        "n_accepted": n_total,
        "n_rejected": rejected_count,
        "class_pairs": class_pairs,
        "style_strength": cc_cfg["style_strength"],
    }

    with open(os.path.join("results", "cue_conflict_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join("results", "cue_conflict_metadata.json"), "w") as f:
        json.dump(accepted_meta, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
