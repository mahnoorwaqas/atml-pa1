# Task 4 — Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknowns)

## Data
CIFAR-10 and CIFAR-100 both download automatically via torchvision
(`download=True`) — no manual setup needed, unlike PACS.

## Build/run order
1. `python3 task4/train.py --config task4/configs/vanilla.yaml`
2. `python3 task4/train.py --config task4/configs/gcsc.yaml`
3. `python3 task4/train.py --config task4/configs/proser.yaml` (needs Vanilla's checkpoint)
4. `python3 task4/extract_outputs.py --config task4/configs/vanilla.yaml`
5. `python3 task4/extract_outputs.py --config task4/configs/gcsc.yaml`
6. `python3 task4/extract_outputs.py --config task4/configs/proser.yaml`
7. `python3 task4/evaluate_osr.py --config task4/configs/vanilla.yaml` — produces both required tables, the score-distribution figure, and failure analysis.

Steps 1–3 are slow (100/100/50 epochs each on CIFAR-10) — budget real GPU time. Steps 4–6 are quick (one forward pass per image, no training).

## PROSER's loss construction — verified against the actual paper
`methods/proser.py` was corrected against Zhou et al. (2021)'s actual equations (Eq. 4–7, Algorithm 1) after the paper PDF was provided. Two real errors in the first draft were caught and fixed:
1. **Architecture**: the paper collapses the C=5 dummy classifiers via `max()` into a single scalar before appending it to the K known logits — always a (K+1)-dimensional output — not a (K+C)-way softmax treating each dummy as its own class, which the first draft used.
2. **Loss target**: with that correction, the "push the dummy to win" term's target is the *fixed* index K+1 (Eq. 5), not a per-example argmax-derived pseudo-label. The masking also sets the true-class entry to exactly 0, per the paper's literal wording — not `-inf`.

The placeholder detection score is now the softmax probability mass on the dummy/"K+1" slot of that same (K+1)-way vector, directly following the paper's framing of PROSER as an ordinary (K+1)-way classifier where class K+1 *is* "unknown" — simpler and better-grounded than the first draft's invented difference-of-maxes formula.

**One deliberate simplification, noted in `proser.py`'s docstring**: the paper's own Sec. 4.3 post-training bias-calibration step (searching a bias added to the dummy logit so 95% of validation data is recognized as known) isn't implemented separately — it's subsumed by the assignment's own uniform 95th-percentile threshold protocol in `evaluate_osr.py`, which already calibrates a threshold on whatever score is used. Doing both would double-calibrate the same thing. Flag this if precise paper fidelity matters for your write-up.


## Other design choices made here that the assignment leaves open (state in your report)
- **Mahalanobis distance** uses ONE shared diagonal covariance pooled across all 10 classes' centered residuals (not per-class), per the assignment's literal wording ("one shared diagonal covariance").

## Not implemented
- **RPL (Step 5, optional extension)** — skipped entirely, consistent with the earlier time-budget triage (cut first when time is tight; not required).
- Score-distribution figure only covers MSP/MLS/Mahalanobis (as the assignment's Required Evidence specifies) — Energy isn't plotted, only tabled.

## Reused across methods
- `methods/vanilla.py`'s `build_model`/`train_cifar_classifier`/`evaluate_accuracy` are reused unchanged by `gcsc.py` — GCSC differs only in its transform (RandAugment inserted by `train.py`, not by any code in `gcsc.py` itself), per the assignment's "exactly the vanilla recipe with one change."
- `models/resnet_cifar.py`'s `forward_features`/`logits_and_dummy` are used identically by all three methods' output extraction, so every score in `scores/` is guaranteed to see the same feature/logit definitions.
