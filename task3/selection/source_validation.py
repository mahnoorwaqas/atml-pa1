"""
Single source of truth for "mean macro-F1 across the three source
validation domains" -- the ONLY signal the assignment permits for
checkpoint selection and hyperparameter choices in Task 3 (never Sketch).
shared/train_utils.py's training loops compute this same quantity inline
during training for early stopping; this module exists so evaluation code
(and anyone auditing for target leakage) has one explicit, obviously-
source-only function to point to.
"""
from shared.train_utils import evaluate_domain


def mean_source_macro_f1(model, source_val_loaders, device):
    f1s = []
    for loader in source_val_loaders.values():
        _, f1 = evaluate_domain(model, loader, device)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def per_domain_and_worst(model, source_val_loaders, device):
    """Returns ({domain: {"accuracy":.., "macro_f1":..}}, worst_domain_name)."""
    results = {}
    for domain_name, loader in source_val_loaders.items():
        acc, f1 = evaluate_domain(model, loader, device)
        results[domain_name] = {"accuracy": acc, "macro_f1": f1}
    worst_domain = min(results, key=lambda d: results[d]["macro_f1"])
    return results, worst_domain
