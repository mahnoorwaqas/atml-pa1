"""
Step 3: SAM (Sharpness-Aware Minimization), standard non-adaptive form
(Foret et al. 2021). rho=0.05 for the main comparison (Step 5's controlled
study sweeps this via configs/sam.yaml's controlled_study section). Every
source batch takes TWO forward/backward passes: an ascent step finds a
worst-case perturbation of radius rho, then the real optimizer step uses
gradients computed AT the perturbed point but applied to the ORIGINAL
parameters. This file inherits whatever train()/eval() + BatchNorm-freeze
state the caller set beforehand -- it doesn't toggle those itself, so both
passes share the same frozen-BN policy as every other Task 2/3 method.
"""
import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from shared.train_utils import evaluate_domain
from task3.models.backbone import build_resnet18, set_bn_eval


def build_model(cfg):
    return build_resnet18(n_classes=cfg["data"]["n_classes"], weights=cfg["model"]["weights"])


def sam_step(model, optimizer, source_images, source_labels, rho):
    """
    One SAM update. Perturbs ALL trainable parameters together using a
    single GLOBAL gradient norm (the standard SAM convention), not a
    per-parameter-tensor norm.
    """
    optimizer.zero_grad()
    loss_at_original = F.cross_entropy(model(source_images), source_labels)
    loss_at_original.backward()

    params = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    scale = rho / (grad_norm + 1e-12)

    epsilons = []
    with torch.no_grad():
        for p in params:
            e = p.grad * scale
            p.add_(e)
            epsilons.append(e)

    optimizer.zero_grad()
    loss_at_perturbed = F.cross_entropy(model(source_images), source_labels)
    loss_at_perturbed.backward()

    with torch.no_grad():
        for p, e in zip(params, epsilons):
            p.sub_(e)  # restore original parameters BEFORE stepping

    optimizer.step()  # updates original params using the perturbed-point gradient

    return {
        "loss_at_original": float(loss_at_original.item()),
        "loss_at_perturbed": float(loss_at_perturbed.item()),
    }


def train_sam(model, batch_iterator, optimizer, rho, source_val_loaders, cfg, checkpoint_path, device, max_steps_per_epoch=None):
    """
    SAM needs its own loop (not shared/train_utils.py::train_loop) since each
    step is two forward/backward passes with manual parameter perturbation,
    not a single compute_loss_fn call -- but epoch/early-stopping/
    checkpointing bookkeeping is otherwise identical to every other method.
    """
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stop_patience"]
    steps_per_epoch = max_steps_per_epoch or cfg["training"].get("steps_per_epoch") or 50

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        set_bn_eval(model)  # frozen BN running stats during BOTH SAM passes

        for _ in (pbar := tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False)):
            source_images, source_labels, _, _ = batch_iterator.next_batch()
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)

            logs = sam_step(model, optimizer, source_images, source_labels, rho)
            pbar.set_postfix({k: f"{v:.3g}" for k, v in logs.items()})
            history.append({**logs, "epoch": epoch})

        val_f1s = {}
        for domain_name, loader in source_val_loaders.items():
            _, f1 = evaluate_domain(model, loader, device)
            val_f1s[domain_name] = f1
        mean_val_f1 = sum(val_f1s.values()) / len(val_f1s)
        print(f"epoch {epoch}: mean_val_f1={mean_val_f1:.4f} ({val_f1s})")

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return history, best_val_f1