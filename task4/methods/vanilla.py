"""
Step 1: Vanilla closed-set baseline. Ten-class CIFAR-ResNet-18, trained from
random initialization with plain crop+flip augmentation.

train_cifar_classifier() here is reused UNCHANGED by gcsc.py -- GCSC only
adds RandAugment to the transform (via cifar10.py's build_train_transform),
per the assignment's "exactly the vanilla recipe with one change."
"""
import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from task4.models.resnet_cifar import build_resnet_cifar


def build_model(n_classes=10):
    return build_resnet_cifar(n_classes=n_classes)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / total


def train_cifar_classifier(model, train_loader, val_loader, device, cfg, checkpoint_path, seed=6304):
    """
    SGD lr=0.1, momentum=0.9, weight_decay=5e-4, cosine decay over
    max_epochs, batch_size set by the caller's DataLoader, seed 6304.
    Checkpoint selection: HIGHEST CIFAR-10 validation accuracy (the
    assignment runs the FULL epoch budget, not early stopping -- it keeps
    whichever epoch's checkpoint was best).
    """
    torch.manual_seed(seed)

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
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.3g}"})

        scheduler.step()

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
