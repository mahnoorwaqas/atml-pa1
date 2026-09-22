"""
Per-class target accuracy changes vs. Source-only, and the dominant
confusions for the classes with the largest improvement/degradation
(Step 5's class-specific negative-transfer check).
"""
import torch


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def per_class_accuracy(model, target_loader, device, n_classes):
    preds, labels = _predict(model, target_loader, device)
    accs = {}
    for c in range(n_classes):
        mask = labels == c
        accs[c] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else None
    return accs, preds, labels


def compare_per_class(source_only_accs, method_accs, class_names, top_k=2):
    """Returns the top_k classes with the largest improvement and degradation
    (method_acc - source_only_acc), by class name."""
    deltas = {
        class_names[c]: method_accs[c] - source_only_accs[c]
        for c in method_accs
        if method_accs[c] is not None and source_only_accs.get(c) is not None
    }
    sorted_deltas = sorted(deltas.items(), key=lambda kv: kv[1])
    largest_degradation = sorted_deltas[:top_k]
    largest_improvement = sorted_deltas[-top_k:][::-1]
    return {"largest_improvement": largest_improvement, "largest_degradation": largest_degradation}


def confusion_for_class(preds, labels, class_idx, class_names):
    """Returns {predicted_class_name: count} for all true examples of class_idx."""
    mask = labels == class_idx
    predicted_for_class = preds[mask]
    counts = {}
    for p in predicted_for_class:
        name = class_names[p]
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
