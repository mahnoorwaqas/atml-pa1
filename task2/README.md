# Task 2 — Unsupervised Domain Adaptation (PACS)

## Data (you must place this yourself)
PACS's official distribution is a gated Google Drive link with no stable
direct-download URL, so nothing here downloads it automatically. Get PACS
from a public mirror and place it at:
```
./data_raw/PACS/<domain>/<class>/*.jpg
```
domains: `photo`, `art_painting`, `cartoon`, `sketch`. Each should contain 7
class folders: dog, elephant, giraffe, guitar, horse, house, person.
Update `data.root` in `configs/base.yaml` if you put it elsewhere.

## Build/run order
1. `python3 task2/train.py --config task2/configs/source_only.yaml` — train first; Task 3 reuses this checkpoint unchanged.
2. `python3 task2/train.py --config task2/configs/dan.yaml`
3. `python3 task2/train.py --config task2/configs/dann.yaml`
4. `python3 task2/train.py --config task2/configs/cdan.yaml`
5. `python3 task2/evaluate_final.py --config task2/configs/base.yaml` — Step 5's comparison table, domain separability, per-class analysis.
6. `python3 task2/run_controlled_study.py --config task2/configs/dan.yaml` **OR** `--config task2/configs/dann.yaml` (Step 6 — pick ONE, per the assignment).

## Design choices made here that the assignment leaves open (state these in your report)
- **"Epoch" for domain-balanced batches**: not fully specified for a protocol drawing fixed-size batches from 3 differently-sized source domains. Defaults to covering the smallest source domain once (`shared/pacs_protocol.py::default_steps_per_epoch`); override `training.steps_per_epoch` in `base.yaml` to fix a specific value instead.
- **DANN/CDAN checkpoint loading at eval time** uses `strict=False` since the saved state dict includes `domain_discriminator.*` weights that the eval-time architecture (built via `build_resnet18`, no discriminator attached) doesn't define — this is expected and harmless; the discriminator is training-only scaffolding.

## Attribution
`shared/mmd.py`'s multi-kernel RBF construction and `task2/models/domain_discriminator.py`'s gradient-reversal layer are original implementations of the mechanisms described in Ganin et al. (2016) and Long et al. (2018) — not vendored from a public repo (unlike Task 1's AdaIN, which is a straight copy of naoto0804/pytorch-AdaIN).

## Not yet implemented here
- Training/alignment-loss curve plotting (Required Evidence item 2) — `train.py` saves a full per-step `history` log to `<checkpoint>_training_log.json`; plotting from that log isn't wired up yet.
- Cue-conflict-style failure-case write-up for Required Evidence item 3 (selected confusions/failure cases) — `evaluate_final.py`'s `per_class_analysis.json` gives you the confusion counts to build this from, but doesn't generate report-ready figures.
