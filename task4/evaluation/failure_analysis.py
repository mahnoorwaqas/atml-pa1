"""
Inspect incorrectly accepted unknowns (score <= threshold) at a given
model/score's threshold -- Required Evidence: "at least three informative
near-unknown failures and three far-unknown failures."
"""
import numpy as np


def incorrectly_accepted(unknown_scores, unknown_predictions, unknown_fine_labels, threshold, class_names, unknown_class_names, n_examples=3, seed=6304):
    """
    unknown_scores: (N,) unknownness scores for one unknown group (near or far).
    unknown_predictions: (N,) predicted CIFAR-10 class index for each.
    unknown_fine_labels: (N,) CIFAR-100 fine-label index for each (from
        cifar100_unknowns.py's dataset -- NOT a CIFAR-10 label).
    class_names: CIFAR-10 class names, indexed by unknown_predictions.
    unknown_class_names: CIFAR-100 class names, indexed by unknown_fine_labels.
    Returns up to n_examples dicts for unknowns with score <= threshold
    (incorrectly accepted as known).
    """
    unknown_scores = np.asarray(unknown_scores)
    unknown_predictions = np.asarray(unknown_predictions)
    unknown_fine_labels = np.asarray(unknown_fine_labels)

    accepted_indices = np.where(unknown_scores <= threshold)[0]

    rng = np.random.RandomState(seed)
    rng.shuffle(accepted_indices)
    selected = accepted_indices[:n_examples]

    return [
        {
            "index": int(i),
            "true_unknown_class": unknown_class_names[unknown_fine_labels[i]],
            "predicted_known_class": class_names[unknown_predictions[i]],
            "score": float(unknown_scores[i]),
            "threshold": float(threshold),
        }
        for i in selected
    ]
