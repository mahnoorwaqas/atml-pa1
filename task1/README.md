# Task 1 — Inductive Biases and Representations

Skeleton only — every `raise NotImplementedError` is a TODO. Fill in order,
since later steps depend on earlier ones (see `scripts/run_task1.py` step order).

## Setup
```
pip install torch torchvision open_clip_torch scikit-learn umap-learn matplotlib pyyaml
```

## Build order
1. `data/make_subset.py` — stratified split + 500-image test subset (do this first, everything reuses these indices)
2. `models/backbones.py` — ResNet-50 / ViT-B/16 / CLIP wrappers + linear head
3. `data/transforms.py` — grayscale, chosen color transform, translation, patch shuffle
4. `data/make_cue_conflicts.py` — AdaIN cue conflicts (needs a public AdaIN checkpoint — decide which implementation to vendor/pip-install before writing this)
5. `analysis/evaluate_bias.py`, `feature_similarity.py`, `representation.py`
6. `scripts/run_task1.py` — wires it all together, run step by step

## Config
All fixed settings (seed 6304, backbone weight versions, epoch budgets) live in
`configs/config.yaml`. Your **experimental-design choices** (dataset, cue-conflict
pairs, style strength, additional color transform, t-SNE vs UMAP + settings) are
marked in the config with inline comments — pick these deliberately and state a
hypothesis for each in the report, per the assignment's requirement.

## Known open decisions (fill in before running)
- Dataset: defaulted to `stl10` in config — switch to `oxford_pets` if preferred
- Additional color transform: defaulted to `hue_rotation` — swap to `palette_transfer` or `class_swapped_stats` if preferred
- AdaIN implementation source: not pinned yet — pick one public repo/checkpoint and note it in this README for attribution
- Cue-conflict rejection rule: must be concrete and defined *before* running any model on the outputs — draft this in `make_cue_conflicts.py` early
