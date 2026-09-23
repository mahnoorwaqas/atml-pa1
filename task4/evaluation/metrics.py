"""Core OSR ranking metric (AUROC) and closed-set classification accuracy."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_known_vs_unknown(known_scores, unknown_scores):
    """
    known_scores, unknown_scores: 1D arrays of unknownness (higher = more
    novel). Positive class = unknown (label 1), known = 0.
    """
    known_scores = np.asarray(known_scores)
    unknown_scores = np.asarray(unknown_scores)
    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def closed_set_accuracy(logits, labels):
    """CSA: top-1 accuracy on known-class logits only."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    return float(np.mean(logits.argmax(axis=1) == labels))
