"""
Step 4: PROSER -- classifier and data placeholders, following Zhou et al.
(2021), "Learning Placeholders for Open-Set Recognition" (CVPR 2021).
Verified against the paper's Eq. 4-7 and Algorithm 1 (previous draft of this
file was an unverified guess -- this one is checked line-by-line against the
actual equations).

Architecture (paper Sec. 4.1, "Learning multiple dummy classifiers"):
  C=5 dummy classifiers are learned (model.dummy_fc: Linear(512, 5)), but
  they are COLLAPSED via max() into a single scalar before being appended
  to the K known logits:
      f_hat(x) = [W^T phi(x), max_k(w_hat_k^T phi(x))]   -- always (K+1)-dim,
  regardless of C. This is `combine_logits()` below.

Classifier placeholders (Eq. 5, first half of each batch):
  l1 = CE(f_hat(x), y) + beta * CE(f_hat(x) with the TRUE-class entry set to
       EXACTLY 0, target=K+1)
  The first term keeps the true class the largest response overall. The
  second term -- with the true class zeroed out (not masked to -inf; the
  paper's literal wording is "set the predicted probability of ground-truth
  ... to 0") -- pushes the dummy slot to be the LARGEST among what remains,
  i.e. the second-largest response overall. beta=1.

Data placeholders (Eq. 6-7, second half of each batch):
  Manifold-mix two different-class examples' layer-2 features (Eq. 6,
  unchanged from the previous draft -- this part was already correct), run
  the mix through the rest of the network, and push the resulting (K+1)-way
  vector toward the FIXED target K+1 (Eq. 7) -- no masking needed here,
  since a mixed instance has no true known-class label to protect.
  l2 = CE(f_hat(x_mixed), target=K+1)
  Total: l_total = l1 + gamma * l2, gamma=0.1 (Algorithm 1, line 8).

NOT implemented: the paper's Sec. 4.3 post-training bias-calibration step
(searching a scalar bias added to the dummy logit so that 95% of validation
data is recognized as known). This is subsumed here by the assignment's own
uniform 95th-percentile threshold protocol (task4/evaluate_osr.py), which
already calibrates a threshold on whatever score is used -- doing the
paper's internal bias search on top would double-calibrate the same thing.
Flag this as a deliberate simplification in your report if precise paper
fidelity matters for your write-up.

Initializes from the selected Vanilla checkpoint, per the assignment.
"""
import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from task4.models.resnet_cifar import (
    build_resnet_cifar, forward_up_to_layer2, forward_from_layer2, logits_and_dummy,
)
from task4.methods.manifold_mixup import manifold_mixup
from task4.methods.vanilla import evaluate_accuracy

N_DUMMY = 5


def build_model(n_classes=10, vanilla_checkpoint_path=None, device="cpu"):
    model = build_resnet_cifar(n_classes=n_classes, n_dummy_classifiers=N_DUMMY)
    if vanilla_checkpoint_path is not None:
        if not os.path.exists(vanilla_checkpoint_path):
            raise FileNotFoundError(
                f"{vanilla_checkpoint_path} not found -- train Vanilla first "
                "(python3 task4/train.py --config task4/configs/vanilla.yaml)."
            )
        # strict=False: the Vanilla checkpoint has no dummy_fc weights (PROSER
        # adds them fresh, randomly initialized) -- expected, not an error.
        state = torch.load(vanilla_checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
    return model


def combine_logits(known_logits, dummy_logits):
    """
    known_logits: (N, K), dummy_logits: (N, C). Collapses the C dummy
    classifiers via max (paper's "multiple dummy classifiers" mechanism) and
    appends the result as one extra column. Returns (N, K+1).
    """
    dummy_max, _ = dummy_logits.max(dim=1, keepdim=True)  # (N, 1)
    return torch.cat([known_logits, dummy_max], dim=1)


def classifier_placeholder_loss(known_logits, dummy_logits, labels, beta=1.0):
    """known_logits: (N, K), dummy_logits: (N, C), labels: (N,) true CIFAR-10 labels."""
    N, K = known_logits.shape
    z = combine_logits(known_logits, dummy_logits)  # (N, K+1)

    loss_correct = F.cross_entropy(z, labels)

    z_masked = z.clone()
    z_masked[torch.arange(N, device=z.device), labels] = 0.0  # zero, per the paper's literal wording -- not -inf
    dummy_target = torch.full((N,), K, dtype=torch.long, device=z.device)  # fixed index K+1 (0-indexed: K)
    loss_dummy = F.cross_entropy(z_masked, dummy_target)

    return loss_correct + beta * loss_dummy


def data_placeholder_loss(mixed_known_logits, mixed_dummy_logits):
    """
    mixed_known_logits: (M, K), mixed_dummy_logits: (M, C), from manifold-
    mixed (different-class) examples. No masking needed -- mixed instances
    have no true known-class label to protect; the target is simply the
    fixed dummy index K+1 for every example.
    """
    M, K = mixed_known_logits.shape
    z = combine_logits(mixed_known_logits, mixed_dummy_logits)
    target = torch.full((M,), K, dtype=torch.long, device=z.device)
    return F.cross_entropy(z, target)


def train_proser(model, train_loader, val_loader, device, cfg, checkpoint_path, seed=6304):
    """
    SGD lr=1e-3, momentum=0.9, weight_decay=5e-4, cosine decay, batch_size
    set by the caller's DataLoader, 50 epochs, seed 6304. beta=1, gamma=0.1.
    Each batch is split in half: first half -> classifier placeholders,
    second half -> manifold-mixup data placeholders. Checkpoint selection:
    highest CIFAR-10 validation accuracy (using ONLY the known-class logits).
    """
    torch.manual_seed(seed)
    beta = cfg["method"].get("beta", 1.0)
    gamma = cfg["method"].get("gamma", 0.1)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["training"]["lr"], momentum=0.9,
        weight_decay=cfg["training"]["weight_decay"],
    )
    max_epochs = cfg["training"]["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(max_epochs):
        model.train()
        for images, labels in (pbar := tqdm(train_loader, desc=f"epoch {epoch}", leave=False)):
            images, labels = images.to(device), labels.to(device)
            N = images.shape[0]
            half = N // 2
            if half == 0:
                continue  # skip a degenerate final batch too small to split

            cls_images, cls_labels = images[:half], labels[:half]
            mix_images, mix_labels = images[half:2 * half], labels[half:2 * half]

            optimizer.zero_grad()

            # --- classifier placeholders (first half) ---
            cls_feat = forward_from_layer2(model, forward_up_to_layer2(model, cls_images))
            cls_known_logits, cls_dummy_logits = logits_and_dummy(model, cls_feat)
            loss_cls = classifier_placeholder_loss(cls_known_logits, cls_dummy_logits, cls_labels, beta=beta)

            # --- data placeholders (second half, manifold mixup after layer2) ---
            mix_h = forward_up_to_layer2(model, mix_images)
            mixed_h, lam, perm = manifold_mixup(mix_h, mix_labels)
            mixed_feat = forward_from_layer2(model, mixed_h)
            mix_known_logits, mix_dummy_logits = logits_and_dummy(model, mixed_feat)
            loss_data = data_placeholder_loss(mix_known_logits, mix_dummy_logits)

            loss = loss_cls + gamma * loss_data
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss_cls": f"{loss_cls.item():.3g}", "loss_data": f"{loss_data.item():.3g}"})

        scheduler.step()

        # uses model.fc only (known logits), via full model() forward, which
        # ignores dummy_fc entirely -- correct for CIFAR-10 val accuracy.
        val_acc = evaluate_accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "val_accuracy": val_acc})
        print(f"epoch {epoch}: val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return history, best_val_acc


def proser_detection_score(known_logits, dummy_logits):
    """
    Placeholder-based unknownness score = the softmax probability mass
    landing on the dummy/"K+1" slot of the combined (K+1)-way vector, per
    the paper's framing of PROSER as an ordinary (K+1)-way classifier where
    class K+1 IS "unknown". Higher = more novel.

    known_logits: (N, K), dummy_logits: (N, C) numpy arrays.
    """
    import numpy as np
    known_logits = np.asarray(known_logits)
    dummy_logits = np.asarray(dummy_logits)

    dummy_max = dummy_logits.max(axis=1, keepdims=True)  # (N, 1) -- same collapse as combine_logits()
    z = np.concatenate([known_logits, dummy_max], axis=1)  # (N, K+1)

    exp = np.exp(z - z.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, -1]  # probability mass on the dummy/unknown slot
