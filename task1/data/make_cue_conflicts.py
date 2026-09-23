"""
Generates cue-conflict images (shape/content = A, texture/style = B, and the
reverse direction) using AdaIN (naoto0804/pytorch-AdaIN, vendored in
task1/models/adain_net.py).

Changes vs. the previous version:
  1. Class pairs are HAND-PICKED via cfg["cue_conflict"]["class_pairs"]
     (names) instead of randomly sampled. Falls back to random sampling only
     if that key is absent.
  2. Rejection rule now has BOTH a lower and an upper bound on edge
     correlation, so under-stylized images (style barely applied) are also
     rejected, not just destroyed ones. Thresholds come from the config.
  3. Edge correlation is stored for every accepted image and summarised in
     the output JSON, so you can tune thresholds from real numbers.
  4. --preview mode renders one content/style pair at several alphas so you
     can choose style_strength by eye.
  5. Metadata records class NAMES as well as indices.

The rejection rule still uses only image statistics (never model output),
so it is defined before any model evaluation.

Prerequisite: run download_adain_weights.py first.

Usage:
    python task1/data/make_cue_conflicts.py --config task1/configs/config.yaml
    python task1/data/make_cue_conflicts.py --config task1/configs/config.yaml --preview
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
from torchvision import transforms as tvt
from torchvision.utils import make_grid, save_image

from task1.data.make_subset import load_dataset, _get_labels
from task1.models import adain_net


def load_adain_model(vgg_weights_path, decoder_weights_path, device="cpu"):
    return adain_net.load_adain_model(vgg_weights_path, decoder_weights_path, device=device)


def resolve_class_pairs(class_names, cc_cfg, seed):
    """
    Return a list of (class_idx_a, class_idx_b) tuples.

    Uses cc_cfg["class_pairs"] (list of [name_a, name_b]) when provided.
    Falls back to seeded random sampling otherwise.
    """
    n_pairs = cc_cfg["n_class_pairs"]
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    if cc_cfg.get("class_pairs"):
        pairs = []
        for name_a, name_b in cc_cfg["class_pairs"]:
            for n in (name_a, name_b):
                if n not in name_to_idx:
                    raise ValueError(
                        f"Class '{n}' in cue_conflict.class_pairs not found. "
                        f"Available classes: {sorted(class_names)}"
                    )
            if name_a == name_b:
                raise ValueError(f"Pair ({name_a}, {name_b}) uses the same class twice.")
            pairs.append((name_to_idx[name_a], name_to_idx[name_b]))
        if len(pairs) < n_pairs:
            raise ValueError(
                f"class_pairs has {len(pairs)} entries but n_class_pairs={n_pairs}."
            )
        return pairs[:n_pairs]

    rng = random.Random(seed)
    all_pairs = list(itertools.combinations(range(len(class_names)), 2))
    rng.shuffle(all_pairs)
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} class pairs but only {len(all_pairs)} exist."
        )
    return all_pairs[:n_pairs]


def stylize(content_tensor, style_tensor, adain_model, alpha):
    """
    content_tensor, style_tensor: (1, 3, H, W) floats in [0, 1] on the same
    device as the model. Returns stylized (1, 3, H, W) clamped to [0, 1].
    """
    vgg_encoder, decoder_net = adain_model
    with torch.no_grad():
        output = adain_net.style_transfer(
            vgg_encoder, decoder_net, content_tensor, style_tensor, alpha=alpha
        )
    return output.clamp(0, 1)


def _to_np(img):
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    return np.asarray(img)


def _edge_map(img):
    gray = img.mean(axis=0)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_correlation(content_image, stylized_image):
    """Pearson correlation between edge maps; None if undefined."""
    content_edges = _edge_map(_to_np(content_image)).flatten()
    stylized_edges = _edge_map(_to_np(stylized_image)).flatten()
    if content_edges.std() == 0 or stylized_edges.std() == 0:
        return None
    corr = np.corrcoef(content_edges, stylized_edges)[0, 1]
    return None if np.isnan(corr) else float(corr)


def passes_rejection_rule(content_image, stylized_image,
                          edge_corr_min=0.30, edge_corr_max=0.90,
                          min_std_threshold=0.02):
    """
    Image-statistics-only rule, defined BEFORE any model prediction is seen.
    Keep the image only if:
      1. It is not near-uniform (pixel std >= min_std_threshold).
      2. Its edge map still correlates with the content's edge map
         (>= edge_corr_min)  -> object silhouette survived.
      3. Its edge map does NOT correlate almost perfectly (<= edge_corr_max)
         -> style was actually applied, image isn't just the content again.
    Returns (keep: bool, corr: float | None).
    """
    stylized = _to_np(stylized_image)
    if stylized.std() < min_std_threshold:
        return False, None

    corr = edge_correlation(content_image, stylized_image)
    if corr is None:
        return False, None
    if corr < edge_corr_min or corr > edge_corr_max:
        return False, corr
    return True, corr


def _load_image_as_tensor(dataset, idx, size=224):
    img, _ = dataset[idx]
    img = img.convert("RGB")
    tf = tvt.Compose([tvt.Resize((size, size)), tvt.ToTensor()])
    return tf(img).unsqueeze(0)


def run_preview(cfg, device, adain_model, train_dataset, class_names, images_by_class, class_pairs, seed):
    """Render one content/style pair at several alphas for visual tuning."""
    alphas = [0.5, 0.65, 0.8, 0.9, 1.0]
    rng = random.Random(seed)
    rows = []
    for (a, b) in class_pairs:
        content_idx = rng.choice(images_by_class[a])
        style_idx = rng.choice(images_by_class[b])
        content = _load_image_as_tensor(train_dataset, content_idx).to(device)
        style = _load_image_as_tensor(train_dataset, style_idx).to(device)
        row = [content[0].cpu(), style[0].cpu()]
        for alpha in alphas:
            row.append(stylize(content, style, adain_model, alpha)[0].cpu())
        rows.extend(row)
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "cue_conflict_alpha_preview.png")
    grid = make_grid(rows, nrow=2 + len(alphas), padding=4)
    save_image(grid, out_path)
    print(f"Saved preview to {out_path}")
    print(f"Columns: content | style | alpha={alphas}")
    print("Rows (content class -> style class): "
          + ", ".join(f"{class_names[a]}->{class_names[b]}" for a, b in class_pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preview", action="store_true",
                        help="Render an alpha sweep for each pair and exit.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    cc_cfg = cfg["cue_conflict"]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    edge_corr_min = float(cc_cfg.get("edge_corr_min", 0.30))
    edge_corr_max = float(cc_cfg.get("edge_corr_max", 0.90))
    min_std = float(cc_cfg.get("min_std_threshold", 0.02))

    train_dataset, _, class_names = load_dataset(cfg)
    labels = _get_labels(train_dataset)
    images_by_class = {c: np.where(labels == c)[0].tolist() for c in range(len(class_names))}

    class_pairs = resolve_class_pairs(class_names, cc_cfg, seed)
    print("Class pairs:", [(class_names[a], class_names[b]) for a, b in class_pairs])

    weights_dir = os.path.join("models", "adain_weights")
    adain_model = load_adain_model(
        os.path.join(weights_dir, "vgg_normalised.pth"),
        os.path.join(weights_dir, "decoder.pth"),
        device=device,
    )

    if args.preview:
        run_preview(cfg, device, adain_model, train_dataset, class_names,
                    images_by_class, class_pairs, seed)
        return

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
    rejected_corrs = []
    accepted_corrs = []

    for content_class, style_class, direction in combos:
        n_accepted = 0
        attempts = 0

        while n_accepted < target_per_combo and attempts < max_attempts_per_combo:
            attempts += 1
            content_idx = rng.choice(images_by_class[content_class])
            style_idx = rng.choice(images_by_class[style_class])

            content_tensor = _load_image_as_tensor(train_dataset, content_idx).to(device)
            style_tensor = _load_image_as_tensor(train_dataset, style_idx).to(device)

            stylized = stylize(content_tensor, style_tensor, adain_model, cc_cfg["style_strength"])

            keep, corr = passes_rejection_rule(
                content_tensor[0], stylized[0],
                edge_corr_min=edge_corr_min,
                edge_corr_max=edge_corr_max,
                min_std_threshold=min_std,
            )

            if keep:
                fname = (f"{class_names[content_class]}_{class_names[style_class]}_"
                         f"{direction}_{n_accepted:03d}.png")
                save_path = os.path.join(out_dir, fname)
                save_image(stylized[0].cpu(), save_path)

                accepted_meta.append({
                    "path": save_path,
                    "content_class": content_class,
                    "style_class": style_class,
                    "content_class_name": class_names[content_class],
                    "style_class_name": class_names[style_class],
                    "direction": direction,
                    "content_idx": content_idx,
                    "style_idx": style_idx,
                    "edge_corr": corr,
                })
                accepted_corrs.append(corr)
                n_accepted += 1
            else:
                rejected_count += 1
                if corr is not None:
                    rejected_corrs.append(corr)

        if n_accepted < target_per_combo:
            print(
                f"WARNING: only {n_accepted}/{target_per_combo} accepted for "
                f"combo (content={class_names[content_class]}, "
                f"style={class_names[style_class]}, {direction}) "
                f"after {attempts} attempts."
            )

    n_total = len(accepted_meta)
    if n_total < cc_cfg["min_valid_conflicts"]:
        print(
            f"WARNING: total accepted conflicts ({n_total}) is below the "
            f"required minimum ({cc_cfg['min_valid_conflicts']})."
        )

    def _stats(vals):
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    summary = {
        "n_accepted": n_total,
        "n_rejected": rejected_count,
        "class_pairs": [[class_names[a], class_names[b]] for a, b in class_pairs],
        "style_strength": cc_cfg["style_strength"],
        "edge_corr_min": edge_corr_min,
        "edge_corr_max": edge_corr_max,
        "accepted_edge_corr_stats": _stats(accepted_corrs),
        "rejected_edge_corr_stats": _stats(rejected_corrs),
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "cue_conflict_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "cue_conflict_metadata.json"), "w") as f:
        json.dump(accepted_meta, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()