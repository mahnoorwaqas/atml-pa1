"""
Step 4's comparison table: per-source-domain accuracy/macro-F1, mean-source
and worst-source values, Sketch accuracy/macro-F1, and the change in Sketch
accuracy relative to ERM.
"""
from shared.train_utils import evaluate_domain


def evaluate_all_domains(model, source_val_loaders, target_loader, device):
    source_results = {}
    for domain_name, loader in source_val_loaders.items():
        acc, f1 = evaluate_domain(model, loader, device)
        source_results[domain_name] = {"accuracy": acc, "macro_f1": f1}

    accuracies = [r["accuracy"] for r in source_results.values()]
    f1s = [r["macro_f1"] for r in source_results.values()]

    sketch_acc, sketch_f1 = evaluate_domain(model, target_loader, device)

    return {
        "per_source_domain": source_results,
        "mean_source_accuracy": sum(accuracies) / len(accuracies),
        "mean_source_macro_f1": sum(f1s) / len(f1s),
        "worst_source_accuracy": min(accuracies),
        "worst_source_macro_f1": min(f1s),
        "sketch_accuracy": sketch_acc,
        "sketch_macro_f1": sketch_f1,
    }


def build_comparison_table(results_by_method):
    if "erm" not in results_by_method:
        raise KeyError("Need an 'erm' entry to compute Sketch-accuracy deltas against.")

    baseline_sketch_acc = results_by_method["erm"]["sketch_accuracy"]

    table = {}
    for method_name, result in results_by_method.items():
        entry = dict(result)
        entry["sketch_accuracy_change_vs_erm"] = result["sketch_accuracy"] - baseline_sketch_acc
        table[method_name] = entry
    return table
