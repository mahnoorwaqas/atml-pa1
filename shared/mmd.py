"""
Multi-kernel MMD (Maximum Mean Discrepancy), shared by Task 2's DAN and
Task 3's DAN-DG. Not part of the assignment's suggested repo listing, but
both methods need the IDENTICAL kernel construction ("use the same MMD
implementation and kernel construction as Task 2"), so this lives in
shared/ rather than being duplicated in task2/ and task3/.
"""
import torch


def _pairwise_sq_dists(X):
    """X: (N, D). Returns (N, N) squared Euclidean distances."""
    sq_norms = (X ** 2).sum(dim=1, keepdim=True)
    return sq_norms + sq_norms.T - 2 * X @ X.T


def _rbf_kernel_sum(X, Y, bandwidth_scales=(0.5, 1.0, 2.0)):
    """
    Sum of three RBF kernels evaluated on the combined [X; Y] batch, with
    bandwidths = bandwidth_scales * (median pairwise squared distance in the
    CURRENT combined batch) -- recomputed every call, per the assignment.
    Returns the full (N_total, N_total) combined kernel matrix.
    """
    combined = torch.cat([X, Y], dim=0)
    sq_dists = _pairwise_sq_dists(combined)

    n = combined.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=combined.device)
    median_sq_dist = sq_dists[off_diag_mask].median()
    median_sq_dist = torch.clamp(median_sq_dist, min=1e-8)  # avoid a degenerate 0-bandwidth kernel

    kernel_sum = torch.zeros_like(sq_dists)
    for scale in bandwidth_scales:
        bandwidth = scale * median_sq_dist
        kernel_sum = kernel_sum + torch.exp(-sq_dists / (2 * bandwidth))

    return kernel_sum


def mmd_squared(X, Y, bandwidth_scales=(0.5, 1.0, 2.0)):
    """
    Squared MMD between samples X (N_x, D) and Y (N_y, D), using the kernel
    trick (never constructs phi explicitly):
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    n_x = X.shape[0]
    K = _rbf_kernel_sum(X, Y, bandwidth_scales)

    K_xx = K[:n_x, :n_x]
    K_yy = K[n_x:, n_x:]
    K_xy = K[:n_x, n_x:]

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
