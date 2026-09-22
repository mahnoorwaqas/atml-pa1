"""
Common local sharpness proxy (Step 4), used identically for ERM, DAN-DG, and
SAM: fix ONE validation batch (32 examples per source domain, seed 6304),
measure the cross-entropy increase after ONE normalized gradient-ascent
perturbation of radius 0.05:
    delta_sharp = L(theta + eps) - L(theta),  eps = 0.05 * grad / ||grad||_2
A standardized LOCAL diagnostic, not a claim about global flatness.
"""
import torch
import torch.nn.functional as F

from task3.models.backbone import set_bn_eval


def build_fixed_sharpness_batch(source_val_loaders, n_per_domain=32, seed=6304):
    """
    Builds ONE fixed batch (identical across every model compared) by taking
    the first n_per_domain examples from each source validation loader.
    `seed` is accepted for interface consistency with the rest of the
    pipeline, though eval loaders are already unshuffled, so this is really
    just a deterministic slice.
    """
    torch.manual_seed(seed)
    images_list, labels_list = [], []
    for domain_name, loader in source_val_loaders.items():
        collected = 0
        for images, labels in loader:
            take = min(n_per_domain - collected, images.shape[0])
            images_list.append(images[:take])
            labels_list.append(labels[:take])
            collected += take
            if collected >= n_per_domain:
                break
        if collected < n_per_domain:
            raise ValueError(
                f"Domain '{domain_name}' validation split has fewer than "
                f"{n_per_domain} examples -- can't build the fixed sharpness batch."
            )
    return torch.cat(images_list, dim=0), torch.cat(labels_list, dim=0)


def sharpness_proxy(model, fixed_images, fixed_labels, device, rho=0.05):
    """
    Returns delta_sharp (float) for `model` on the fixed batch. Restores the
    model's ORIGINAL parameters afterward (perturbs and un-perturbs in
    place), so it's safe to call repeatedly on the same model.
    """
    model.eval()
    set_bn_eval(model)  # eval() already covers BN, but explicit for clarity/consistency

    images = fixed_images.to(device)
    labels = fixed_labels.to(device)

    model.zero_grad()
    loss_original = F.cross_entropy(model(images), labels)
    loss_original.backward()

    params = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    scale = rho / (grad_norm + 1e-12)

    epsilons = []
    with torch.no_grad():
        for p in params:
            e = p.grad * scale
            p.add_(e)
            epsilons.append(e)

    with torch.no_grad():
        loss_perturbed = F.cross_entropy(model(images), labels)

    with torch.no_grad():
        for p, e in zip(params, epsilons):
            p.sub_(e)  # restore original parameters

    model.zero_grad()

    return float(loss_perturbed.item() - loss_original.item())
