# Task 3 — Domain Generalization (PACS, Sketch unseen)

## Prerequisite
Train Task 2's `source_only` method first — this is reused UNCHANGED as
Task 3's ERM baseline:
```
python3 task2/train.py --config task2/configs/source_only.yaml
```

## Data
Same PACS copy as Task 2 (`shared/pacs_protocol.py::build_source_splits`
reuses Task 2's saved split file directly, so both tasks see identical
source train/val splits). See `task2/README.md` for how to obtain PACS.

## Build/run order
1. `python3 task3/train.py --config task3/configs/erm.yaml` — just verifies Task 2's checkpoint exists; trains nothing.
2. `python3 task3/train.py --config task3/configs/dan_dg.yaml`
3. `python3 task3/train.py --config task3/configs/sam.yaml`
4. `python3 task3/evaluate_sketch.py --config task3/configs/base.yaml` — Step 4's comparison table, source-domain separability, sharpness proxy, per-class analysis. Sketch is loaded ONLY in this script.
5. `python3 task3/run_controlled_study.py --config task3/configs/dan_dg.yaml` **OR** `--config task3/configs/sam.yaml` (Step 5 — pick ONE). Deliberately Sketch-free.

## Design choices made here that the assignment leaves open (state these in your report)
- **"Epoch" convention** reused unchanged from Task 2 (`shared/pacs_protocol.py::default_steps_per_epoch` — covers the smallest source domain once).
- **SAM's perturbation** uses a single GLOBAL gradient norm across all trainable parameters (the standard SAM convention), not a per-layer norm.
- **The fixed sharpness batch** (32 examples/domain, seed 6304) is built by taking each source validation loader's first 32 examples in iteration order, since eval loaders are already unshuffled — a deterministic slice rather than an explicit re-sample.
- **`run_controlled_study.py` never loads Sketch**, even though the assignment permits "analysis only" Sketch results for interpreting the study — kept deliberately separate so the sweep script itself can't be a vector for target leakage. Evaluate swept checkpoints against Sketch as a separate step afterward, if you want those numbers.

## Reused from Task 2 (not duplicated)
- `models/backbone.py`, `models/classifier_head.py` — thin re-exports of `task2.models.*`, per the assignment's "exactly the same ResNet-18 initialization ... classifier head."
- `shared/mmd.py` — identical MMD kernel construction as Task 2's DAN, per the assignment's explicit requirement.
- `shared/train_utils.py::train_loop` — reused directly for DAN-DG (single-loss method, same shape as Task 2's methods). SAM needed its own loop (`methods/sam.py::train_sam`) since each step is two forward/backward passes, not one.
- `task2/evaluation/class_analysis.py` — per-class accuracy/confusion helpers, reused as-is for the Sketch per-class analysis.

## Not yet implemented here
- Training-curve plotting (Required Evidence item 3: classification loss + MMD penalty curves) — `train.py`/`run_controlled_study.py` save full per-step `history` logs to `*_training_log.json`; plotting from that log isn't wired up yet.
- Turning `per_class_analysis.json`'s confusion data into report-ready failure-case write-ups (Required Evidence item 5).
