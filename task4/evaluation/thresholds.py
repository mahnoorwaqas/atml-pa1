"""
Validation-calibrated rejection: threshold = 95th percentile of unknownness
on KNOWN CIFAR-10 validation data (accepts ~95% of known validation
examples by construction). FPR@95TPR = fraction of unknowns incorrectly
accepted under this convention.
"""
import numpy as np


def calibrate_threshold(val_known_scores, percentile=95.0):
    return float(np.percentile(np.asarray(val_known_scores), percentile))


def acceptance_and_rejection_rates(test_known_scores, near_unknown_scores, far_unknown_scores, threshold):
    """
    accept x when u(x) <= threshold. Returns CIFAR-10 test acceptance rate,
    near/far/all unknown rejection rates, and the corresponding
    FPR@95TPR-style acceptance rates for unknowns (1 - rejection rate).
    """
    test_known_scores = np.asarray(test_known_scores)
    near_unknown_scores = np.asarray(near_unknown_scores)
    far_unknown_scores = np.asarray(far_unknown_scores)
    all_unknown_scores = np.concatenate([near_unknown_scores, far_unknown_scores])

    known_acceptance_rate = float(np.mean(test_known_scores <= threshold))
    near_rejection_rate = float(np.mean(near_unknown_scores > threshold))
    far_rejection_rate = float(np.mean(far_unknown_scores > threshold))
    all_rejection_rate = float(np.mean(all_unknown_scores > threshold))

    return {
        "threshold": threshold,
        "known_acceptance_rate": known_acceptance_rate,
        "near_unknown_rejection_rate": near_rejection_rate,
        "far_unknown_rejection_rate": far_rejection_rate,
        "all_unknown_rejection_rate": all_rejection_rate,
        "near_fpr_at_95tpr": 1.0 - near_rejection_rate,
        "far_fpr_at_95tpr": 1.0 - far_rejection_rate,
        "all_fpr_at_95tpr": 1.0 - all_rejection_rate,
    }
