"""u_MLS(x) = -max_k z_k(x). Retains absolute logit magnitude (unlike MSP's normalized confidence)."""
import numpy as np


def mls_score(logits):
    logits = np.asarray(logits)
    return -logits.max(axis=1)
