"""
Generic Task 2 / Task 3 training loop: handles epoch/step bookkeeping,
BatchNorm-eval enforcement, mean-source-validation-macro-F1 early stopping,
and checkpointing -- identical across source_only/dan/dann/cdan (and Task
3's erm/dan_dg/sam). Only what LOSS each method computes differs (passed in
as compute_loss_fn).
"""
import copy
import os

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from task2.models.backbone import set_bn_eval


@torch.no_grad()
def evaluate_domain(model, loader, device):
    """Returns (accuracy, macro_f1) for one domain's eval loader."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = float((all_preds == all_labels).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    return accuracy, macro_f1


def train_loop(
    model, batch_iterator, optimizer, compute_loss_fn,
    source_val_loaders, cfg, checkpoint_path, device, max_steps_per_epoch=None,
):
    """
    model: the ResNet-18 (with .fc, and .domain_discriminator if DANN/CDAN).
    batch_iterator: a pacs_protocol.DomainBalancedBatchIterator.
    compute_loss_fn(model, source_images, source_labels, source_domain_ids,
                     target_images, step_progress) -> (loss, logs: dict).
        target_images is None for source_only / Task 3 methods.
        step_progress in [0, 1] -- used by DANN/CDAN's GRL alpha schedule.
    source_val_loaders: {domain_name: DataLoader} (EVAL_TRANSFORM, val split).
    Returns (history: list of per-step log dicts, best_val_macro_f1: float).
    """
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stop_patience"]
    steps_per_epoch = max_steps_per_epoch or cfg["training"].get("steps_per_epoch") or 50

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        set_bn_eval(model)  # freeze BN running stats, keep gamma/beta trainable

        for _ in (pbar := tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False)):
            source_images, source_labels, source_domain_ids, target_images = batch_iterator.next_batch()
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            source_domain_ids = source_domain_ids.to(device)
            if target_images is not None:
                target_images = target_images.to(device)

            step_progress = global_step / max(total_steps - 1, 1)

            optimizer.zero_grad()
            loss, logs = compute_loss_fn(
                model, source_images, source_labels, source_domain_ids,
                target_images, step_progress,
            )
            loss.backward()
            # Not in the assignment's spec, but necessary in practice: DANN/
            # CDAN's reversed gradient, combined with the mandatory frozen-
            # BatchNorm-statistics policy (which removes the usual
            # activation-renormalization safety net), can drive weights into
            # a runaway feedback loop -- loss values growing into the
            # thousands within a single epoch. Clipping caps that without
            # changing source_only/DAN's already-stable training at all
            # (their gradients rarely approach this norm anyway). Document
            # this as a practical stabilization choice in your report.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logs["epoch"] = epoch
            logs["step"] = global_step
            logs["total_loss"] = float(loss.item())

            # Live postfix -- shows whatever compute_loss_fn returned, so
            # DANN/CDAN's alpha/domain_loss are visible in real time (not
            # just after the fact from the saved history log), while
            # source_only/DAN show their own (fewer) fields automatically.
            pbar.set_postfix({k: f"{v:.3g}" if isinstance(v, float) else v
                               for k, v in logs.items() if k not in ("epoch", "step")})
            history.append(logs)
            global_step += 1

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