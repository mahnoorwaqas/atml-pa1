"""u_Energy(x) = -log sum_k exp(z_k(x)). Uses ALL logits (unlike MSP/MLS's max-only)."""
import numpy as np
from scipy.special import logsumexp


def energy_score(logits):
    logits = np.asarray(logits)
    return -logsumexp(logits, axis=1)
