"""u_MSP(x) = 1 - max_k p_k(x). Normalized-confidence novelty score."""
import numpy as np


def msp_score(logits):
    """logits: (N, C) raw logits. Returns (N,) unknownness -- higher = more novel."""
    logits = np.asarray(logits)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)
