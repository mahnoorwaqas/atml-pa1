"""
Top-level orchestration for Task 1. Run each step in order (per the assignment,
steps must be completed in order: clean baseline -> color -> shape/texture ->
translation -> patch structure -> representation analysis).

Run from the REPO ROOT (the folder containing task1/), e.g.:
    python task1/scripts/run_task1.py --config task1/configs/config.yaml --step all
    python task1/scripts/run_task1.py --config task1/configs/config.yaml --step clean_baseline

The sys.path shim below also makes `python -m task1.scripts.run_task1 ...` work
from the repo root without it, if you prefer that invocation style.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms as tvt
from tqdm import tqdm

from task1.data.make_subset import load_dataset, _get_labels
from task1.data import transforms as T
from task1.analysis import feature_similarity as fs
from task1.analysis import representation as rep
from task1.models.backbones import get_backbone, LinearHead
from task1.analysis import evaluate_bias as eb

RESULTS_DIR = "results"
HEADS_DIR = os.path.join(RESULTS_DIR, "heads")
FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
BACKBONE_NAMES = ["resnet50", "vit_b_16", "clip_vit_b_32"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_subset_ids(cfg):
    with open(cfg["subset"]["ids_path"]) as f:
        return json.load(f)


def _get_pil_images(dataset, indices, size=224, pil_transform=None, desc="loading images"):
    """
    Load raw images for the given dataset indices, resize to size x size, apply
    an optional PIL-level intervention (e.g. T.to_grayscale), and return a
    list of PIL.Image (mode "RGB").
    """
    resize = tvt.Resize((size, size))
    images = []
    for idx in tqdm(indices, desc=desc, leave=False):
        img, _ = dataset[idx]
        img = resize(img.convert("RGB"))
        if pil_transform is not None:
            img = pil_transform(img)
        images.append(img)
    return images


def _pil_batch_to_tensor(pil_images):
    """List of PIL RGB images (already resized) -> (N,3,H,W) float tensor in [0,1]."""
    to_tensor = tvt.ToTensor()
    return torch.stack([to_tensor(img) for img in pil_images])


def _get_labels_for(dataset, indices):
    labels = _get_labels(dataset)
    return torch.tensor([int(labels[i]) for i in indices], dtype=torch.long)


def _extract_features_batched(backbone, images_tensor, batch_size=64, desc="extracting features"):
    """Runs the frozen backbone over images_tensor in chunks so a 500+ image
    subset doesn't need to fit through the model in one forward pass."""
    feats = []
    n_batches = (images_tensor.shape[0] + batch_size - 1) // batch_size
    for i in tqdm(range(0, images_tensor.shape[0], batch_size), total=n_batches, desc=desc, leave=False):
        batch = images_tensor[i:i + batch_size]
        feats.append(backbone.extract_features(batch).cpu())
    return torch.cat(feats, dim=0)


def _save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


def _load_tensor(path):
    return torch.load(path)


def _load_head(name, backbone, n_classes):
    head = LinearHead(backbone.feature_dim, n_classes)
    state = _load_tensor(os.path.join(HEADS_DIR, f"{name}_head.pt"))
    head.model.load_state_dict(state)
    return head


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_clean_baseline(cfg):
    """
    1. Load subset/split indices (from make_subset.py output).
    2. For each backbone: extract features, train a linear head, get logits
       on the 500-image test subset. For CLIP, also get zero-shot logits.
    3. Compute + save clean metrics (accuracy, macro-F1, mean max confidence)
       per backbone to results/.
    Also caches features/logits/trained heads to disk so later steps don't
    need to retrain or recompute anything from this step.
    """
    ids = _load_subset_ids(cfg)
    train_dataset, test_dataset, class_names = load_dataset(cfg)

    train_idx = ids["train_indices"]
    val_idx = ids["val_indices"]
    test_idx = ids["test_subset_indices"]

    train_labels = _get_labels_for(train_dataset, train_idx)
    val_labels = _get_labels_for(train_dataset, val_idx)
    test_labels = _get_labels_for(test_dataset, test_idx)
    _save_tensor(test_labels, os.path.join(FEATURES_DIR, "test_labels.pt"))

    train_images = _pil_batch_to_tensor(_get_pil_images(train_dataset, train_idx))
    val_images = _pil_batch_to_tensor(_get_pil_images(train_dataset, val_idx))
    test_images = _pil_batch_to_tensor(_get_pil_images(test_dataset, test_idx))
    _save_tensor(test_images, os.path.join(FEATURES_DIR, "clean_test_images.pt"))

    results = {}

    for name in tqdm(BACKBONE_NAMES, desc="backbones"):
        backbone = get_backbone(name, cfg)

        train_feats = _extract_features_batched(backbone, train_images)
        val_feats = _extract_features_batched(backbone, val_images)
        test_feats = _extract_features_batched(backbone, test_images)
        _save_tensor(test_feats, os.path.join(FEATURES_DIR, f"{name}_clean_test_features.pt"))

        head = LinearHead(backbone.feature_dim, len(class_names))
        fit_info = head.fit(
            train_feats.to(backbone.device), train_labels.to(backbone.device),
            val_feats.to(backbone.device), val_labels.to(backbone.device), cfg,
        )
        _save_tensor(head.model.state_dict(), os.path.join(HEADS_DIR, f"{name}_head.pt"))

        test_logits = head.predict_logits(test_feats.to(backbone.device)).cpu()
        _save_tensor(test_logits, os.path.join(FEATURES_DIR, f"{name}_clean_test_logits.pt"))

        results[name] = {**eb.compute_clean_metrics(test_logits, test_labels), **fit_info}

        if name == "clip_vit_b_32":
            # Raw (pre-softmax) similarities — compute_clean_metrics applies its
            # own softmax internally, so DON'T pre-softmax these yourself here,
            # or confidence gets computed from a softmax-of-a-softmax.
            zs_sims = backbone.zero_shot_logits(
                test_images.to(backbone.device), class_names,
                cfg["clip_zeroshot"]["prompt_template"],
            ).cpu()
            _save_tensor(zs_sims, os.path.join(FEATURES_DIR, "clip_zeroshot_test_similarities.pt"))
            results["clip_zeroshot"] = eb.compute_clean_metrics(zs_sims, test_labels)

    with open(os.path.join(RESULTS_DIR, "clean_baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def step_color_bias(cfg):
    """Grayscale + chosen additional color transform, all backbones, vs. clean baseline."""
    ids = _load_subset_ids(cfg)
    _, test_dataset, class_names = load_dataset(cfg)
    test_idx = ids["test_subset_indices"]
    test_labels = _load_tensor(os.path.join(FEATURES_DIR, "test_labels.pt"))

    color_cfg = cfg["color_bias"]
    extra_name = color_cfg["extra_transform"]

    interventions = {"grayscale": T.to_grayscale}
    if extra_name == "hue_rotation":
        degrees = color_cfg["hue_rotation_degrees"]
        interventions["hue_rotation"] = lambda img: T.hue_rotation(img, degrees)
    elif extra_name in ("palette_transfer", "class_swapped_stats"):
        raise NotImplementedError(
            f"'{extra_name}' needs a reference image / precomputed per-class "
            "stats wired up before this step will run — see transforms.py."
        )
    else:
        raise ValueError(f"Unknown color_bias.extra_transform: {extra_name}")

    results = {}
    for intervention_name, pil_fn in interventions.items():
        images = _get_pil_images(test_dataset, test_idx, pil_transform=pil_fn)
        images_tensor = _pil_batch_to_tensor(images)
        _save_tensor(images_tensor, os.path.join(FEATURES_DIR, f"{intervention_name}_test_images.pt"))

        for name in tqdm(BACKBONE_NAMES, desc=f"backbones ({intervention_name})"):
            results.setdefault(intervention_name, {})[name] = eb.compute_intervention_metrics(
                clean_logits, logits, test_labels
            )

        # Also evaluate CLIP zero-shot under this intervention, for completeness.
        backbone = get_backbone("clip_vit_b_32", cfg)
        zs_sims = backbone.zero_shot_logits(
            images_tensor.to(backbone.device), class_names,
            cfg["clip_zeroshot"]["prompt_template"],
        ).cpu()
        clean_zs_sims = _load_tensor(os.path.join(FEATURES_DIR, "clip_zeroshot_test_similarities.pt"))
        results[intervention_name]["clip_zeroshot"] = eb.compute_intervention_metrics(
            clean_zs_sims, zs_sims, test_labels
        )

    with open(os.path.join(RESULTS_DIR, "color_bias_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def step_shape_vs_texture(cfg):
    """
    Requires make_cue_conflicts.py to have already been run AND fully
    implemented (it's currently still a stub — see
    task1/data/make_cue_conflicts.py). Evaluate all backbones on accepted
    cue-conflict images; classify predictions; compute shape bias + coverage.
    """
    meta_path = os.path.join(RESULTS_DIR, "cue_conflict_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Finish implementing and run "
            "make_cue_conflicts.py before this step will work."
        )

    with open(meta_path) as f:
        cue_conflict_meta = json.load(f)

    resize = tvt.Resize((224, 224))
    to_tensor = tvt.ToTensor()
    images_tensor = torch.stack([
        to_tensor(resize(__import__("PIL.Image", fromlist=["Image"]).open(m["path"]).convert("RGB")))
        for m in cue_conflict_meta
    ])
    content_labels = torch.tensor([m["content_class"] for m in cue_conflict_meta], dtype=torch.long)
    style_labels = torch.tensor([m["style_class"] for m in cue_conflict_meta], dtype=torch.long)

    _, _, class_names = load_dataset(cfg)

    results = {}
    for name in tqdm(BACKBONE_NAMES, desc="backbones"):
        backbone = get_backbone(name, cfg)
        head = _load_head(name, backbone, len(class_names))

        feats = _extract_features_batched(backbone, images_tensor)
        logits = head.predict_logits(feats.to(backbone.device)).cpu()
        preds = logits.argmax(dim=1)

        classifications, counts = eb.classify_cue_conflict_predictions(
            preds, content_labels, style_labels
        )
        bias = eb.shape_bias_and_coverage(
            counts["n_shape"], counts["n_texture"], len(cue_conflict_meta)
        )
        results[name] = {**counts, **bias}

    with open(os.path.join(RESULTS_DIR, "shape_vs_texture_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def step_translation(cfg):
    """4 offsets x 4 directions, all backbones; produce translation curve."""
    ids = _load_subset_ids(cfg)
    _, test_dataset, class_names = load_dataset(cfg)
    test_idx = ids["test_subset_indices"]
    test_labels = _load_tensor(os.path.join(FEATURES_DIR, "test_labels.pt"))

    trans_cfg = cfg["translation"]
    offsets = trans_cfg["offsets_px"]
    directions = trans_cfg["directions"]
    padding_mode = trans_cfg["padding_mode"]

    results = {}
    for name in tqdm(BACKBONE_NAMES, desc="backbones"):
        backbone = get_backbone(name, cfg)
        head = _load_head(name, backbone, len(class_names))
        clean_logits = _load_tensor(os.path.join(FEATURES_DIR, f"{name}_clean_test_logits.pt"))

        logits_by_offset = {}
        for offset in tqdm(offsets, desc=f"{name}: offsets", leave=False):
            per_direction_logits = []
            for direction in tqdm(directions, desc=f"offset={offset}px", leave=False):
                pil_fn = lambda img, o=offset, d=direction: T.translate(img, o, d, padding_mode)
                images = _get_pil_images(test_dataset, test_idx, pil_transform=pil_fn)
                images_tensor = _pil_batch_to_tensor(images)
                feats = _extract_features_batched(backbone, images_tensor)
                logits = head.predict_logits(feats.to(backbone.device)).cpu()
                per_direction_logits.append(logits)
            logits_by_offset[offset] = per_direction_logits

        results[name] = eb.translation_curve(clean_logits, logits_by_offset, test_labels)

    with open(os.path.join(RESULTS_DIR, "translation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def step_patch_structure(cfg):
    """4x4 grid shuffle (seed 6304), all backbones; accuracy drop + consistency."""
    ids = _load_subset_ids(cfg)
    _, test_dataset, class_names = load_dataset(cfg)
    test_idx = ids["test_subset_indices"]
    test_labels = _load_tensor(os.path.join(FEATURES_DIR, "test_labels.pt"))

    grid = tuple(cfg["patch_shuffle"]["grid"])
    seed = cfg["seed"]
    pil_fn = lambda img: T.patch_shuffle(img, grid=grid, seed=seed)

    images = _get_pil_images(test_dataset, test_idx, pil_transform=pil_fn)
    images_tensor = _pil_batch_to_tensor(images)
    _save_tensor(images_tensor, os.path.join(FEATURES_DIR, "patch_shuffle_test_images.pt"))

    results = {}
    for name in tqdm(BACKBONE_NAMES, desc="backbones"):
        backbone = get_backbone(name, cfg)
        head = _load_head(name, backbone, len(class_names))

        feats = _extract_features_batched(backbone, images_tensor)
        logits = head.predict_logits(feats.to(backbone.device)).cpu()

        clean_logits = _load_tensor(os.path.join(FEATURES_DIR, f"{name}_clean_test_logits.pt"))
        results[name] = eb.compute_intervention_metrics(clean_logits, logits, test_labels)

    with open(os.path.join(RESULTS_DIR, "patch_structure_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


def step_representation_analysis(cfg):
    """
    Compute cosine stability and joint t-SNE/UMAP projections for:
    grayscale, cue_conflict, translation, and patch_shuffle.

    Translation uses the largest configured offset, with features averaged
    across the four cardinal directions.

    Cue conflicts are paired with the clean content images that were actually
    used to generate each stylized image.
    """
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    from task1.analysis import feature_similarity as fs
    from task1.analysis import representation as rep

    ids = _load_subset_ids(cfg)
    train_dataset, test_dataset, class_names = load_dataset(cfg)
    test_idx = ids["test_subset_indices"]
    test_labels = _load_tensor(
        os.path.join(FEATURES_DIR, "test_labels.pt")
    ).numpy()

    rep_cfg = cfg["representation_analysis"]
    method = rep_cfg["method"]
    method_kwargs = (
        {"perplexity": rep_cfg["tsne_perplexity"]}
        if method == "tsne"
        else {"n_neighbors": rep_cfg["umap_n_neighbors"]}
    )

    meta_path = os.path.join(
        RESULTS_DIR, "cue_conflict_metadata.json"
    )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found -- run make_cue_conflicts.py first."
        )

    with open(meta_path) as f:
        cue_meta = json.load(f)

    fig_dir = os.path.join(RESULTS_DIR, "representation_plots")
    os.makedirs(fig_dir, exist_ok=True)

    resize = tvt.Resize((224, 224))
    to_tensor = tvt.ToTensor()

    # Reuse images already generated by previous stages
    gray_images = _load_tensor(
        os.path.join(FEATURES_DIR, "grayscale_test_images.pt")
    )
    shuffled_images = _load_tensor(
        os.path.join(FEATURES_DIR, "patch_shuffle_test_images.pt")
    )

    # Prepare cue-conflict images and their actual clean content images
    cue_images = torch.stack([
        to_tensor(
            resize(
                Image.open(m["path"]).convert("RGB")
            )
        )
        for m in cue_meta
    ])

    cue_content_images = torch.stack([
        to_tensor(
            resize(
                train_dataset[m["content_idx"]][0].convert("RGB")
            )
        )
        for m in cue_meta
    ])

    stability_results = {}

    for name in tqdm(
        BACKBONE_NAMES,
        desc="Representation analysis",
        unit="backbone",
    ):
        backbone = get_backbone(name, cfg)

        # Reuse clean features from the earlier clean-baseline step
        clean_feats = _load_tensor(
            os.path.join(
                FEATURES_DIR,
                f"{name}_clean_test_features.pt",
            )
        )

        condition_feats = {}

        # Grayscale
        condition_feats["grayscale"] = _extract_features_batched(
            backbone,
            gray_images,
        )

        # Patch shuffle
        condition_feats["patch_shuffle"] = _extract_features_batched(
            backbone,
            shuffled_images,
        )

        # Translation: largest offset, average four directions
        offset = cfg["translation"]["offsets_px"][-1]
        direction_feats = []

        for direction in tqdm(
            cfg["translation"]["directions"],
            desc=f"{name} translation",
            unit="direction",
            leave=False,
        ):
            images = _get_pil_images(
                test_dataset,
                test_idx,
                pil_transform=lambda img, o=offset, d=direction:
                    T.translate(
                        img,
                        o,
                        d,
                        cfg["translation"]["padding_mode"],
                    ),
            )

            direction_feats.append(
                _extract_features_batched(
                    backbone,
                    _pil_batch_to_tensor(images),
                )
            )

        condition_feats["translation"] = torch.stack(
            direction_feats
        ).mean(dim=0)

        # Cue conflict
        cue_clean_feats = _extract_features_batched(
            backbone,
            cue_content_images,
        )

        cue_transformed_feats = _extract_features_batched(
            backbone,
            cue_images,
        )

        # Cosine stability
        stability_results[name] = {
            "grayscale": fs.cosine_stability(
                clean_feats,
                condition_feats["grayscale"],
            ),
            "patch_shuffle": fs.cosine_stability(
                clean_feats,
                condition_feats["patch_shuffle"],
            ),
            "translation": fs.cosine_stability(
                clean_feats,
                condition_feats["translation"],
            ),
            "cue_conflict": fs.cosine_stability(
                cue_clean_feats,
                cue_transformed_feats,
            ),
        }

        # t-SNE / UMAP for test-subset interventions
        for intervention in tqdm(
            ["grayscale", "translation", "patch_shuffle"],
            desc=f"{name} projections",
            unit="plot",
            leave=False,
        ):
            transformed = condition_feats[intervention]

            combined = torch.cat(
                [clean_feats, transformed],
                dim=0,
            ).numpy()

            coords = rep.fit_projection(
                combined,
                method=method,
                seed=cfg["seed"],
                **method_kwargs,
            )

            combined_labels = np.concatenate(
                [test_labels, test_labels]
            )

            condition_labels = (
                ["clean"] * len(test_labels)
                + [intervention] * len(test_labels)
            )

            rep.plot_projection(
                coords,
                combined_labels,
                condition_labels,
                title=f"{name}: clean vs {intervention} ({method})",
                save_path=os.path.join(
                    fig_dir,
                    f"{name}_{intervention}_{method}.png",
                ),
            )

        # t-SNE / UMAP for cue conflicts
        combined_cc = torch.cat(
            [cue_clean_feats, cue_transformed_feats],
            dim=0,
        ).numpy()

        coords_cc = rep.fit_projection(
            combined_cc,
            method=method,
            seed=cfg["seed"],
            **method_kwargs,
        )

        cc_labels = np.array([
            m["content_class"] for m in cue_meta
        ])

        rep.plot_projection(
            coords_cc,
            np.concatenate([cc_labels, cc_labels]),
            ["clean"] * len(cue_meta)
            + ["cue_conflict"] * len(cue_meta),
            title=f"{name}: clean vs cue_conflict ({method})",
            save_path=os.path.join(
                fig_dir,
                f"{name}_cue_conflict_{method}.png",
            ),
        )

    output_path = os.path.join(
        RESULTS_DIR,
        "representation_stability_metrics.json",
    )

    with open(output_path, "w") as f:
        json.dump(stability_results, f, indent=2)

    print(json.dumps(stability_results, indent=2))
    return stability_results


STEPS = {
    "clean_baseline": step_clean_baseline,
    "color_bias": step_color_bias,
    "shape_vs_texture": step_shape_vs_texture,
    "translation": step_translation,
    "patch_structure": step_patch_structure,
    "representation_analysis": step_representation_analysis,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", default="all", choices=list(STEPS.keys()) + ["all"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.step == "all":
        for name, fn in STEPS.items():
            print(f"--- {name} ---")
            fn(cfg)
    else:
        STEPS[args.step](cfg)


if __name__ == "__main__":
    main()