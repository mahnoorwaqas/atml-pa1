"""
Builds the required comparison table: Source-only, DAN, DANN, CDAN across
each source validation domain, mean/target accuracy+macro-F1, and target
accuracy change vs. Source-only. (Domain separability is added separately
by evaluate_final.py, via domain_separability.py.)
"""
from shared.train_utils import evaluate_domain


def evaluate_all_domains(model, source_val_loaders, target_loader, device):
    """Returns per-source-domain metrics, their mean, and target metrics."""
    source_results = {}
    for domain_name, loader in source_val_loaders.items():
        acc, f1 = evaluate_domain(model, loader, device)
        source_results[domain_name] = {"accuracy": acc, "macro_f1": f1}

    target_acc, target_f1 = evaluate_domain(model, target_loader, device)

    mean_source_acc = sum(r["accuracy"] for r in source_results.values()) / len(source_results)
    mean_source_f1 = sum(r["macro_f1"] for r in source_results.values()) / len(source_results)

    return {
        "per_source_domain": source_results,
        "mean_source_accuracy": mean_source_acc,
        "mean_source_macro_f1": mean_source_f1,
        "target_accuracy": target_acc,
        "target_macro_f1": target_f1,
    }


def build_comparison_table(results_by_method):
    """
    results_by_method: {"source_only": evaluate_all_domains(...), "dan": ..., ...}.
    Adds target_accuracy_change_vs_source_only to every entry.
    """
    if "source_only" not in results_by_method:
        raise KeyError("Need a 'source_only' entry to compute accuracy deltas against.")

    baseline_target_acc = results_by_method["source_only"]["target_accuracy"]

    table = {}
    for method_name, result in results_by_method.items():
        entry = dict(result)
        entry["target_accuracy_change_vs_source_only"] = result["target_accuracy"] - baseline_target_acc
        table[method_name] = entry

    return table
