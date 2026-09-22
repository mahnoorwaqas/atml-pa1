"""
Common interventions applied identically across all three backbones, on a
common 224x224 RGB image, BEFORE each model's own normalization.

Implemented with PIL + numpy only (no torch dependency here) so these can be
generated, saved, and inspected independently of which backbone will later
consume them. Every function takes a PIL.Image (mode "RGB") and returns a
PIL.Image (mode "RGB") of the same size.
"""
import random

import numpy as np
from PIL import Image


def to_grayscale(image):
    """Convert to grayscale, then replicate to 3 channels (preserves input shape/mode)."""
    return image.convert("L").convert("RGB")


def hue_rotation(image, degrees):
    """
    Fixed hue rotation in HSV space by `degrees` (0-360).
    Preserves: geometry, saturation, value (luminance/shape cues).
    Changes: hue only (chromatic identity of colors).
    """
    hsv = np.array(image.convert("HSV"), dtype=np.int16)
    shift = int(round(degrees / 360.0 * 256.0))
    hsv[..., 0] = (hsv[..., 0] + shift) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def palette_transfer(image, reference_image):
    """
    Alternative to hue_rotation — transfer per-channel RGB color statistics
    (mean, std) from a reference image onto `image`, preserving image's own
    geometry/content but shifting its overall color palette toward the
    reference's. Only use if this is your chosen additional color intervention.
    """
    img = np.array(image).astype(np.float32)
    ref = np.array(reference_image).astype(np.float32)

    out = np.empty_like(img)
    for c in range(3):
        img_mean, img_std = img[..., c].mean(), img[..., c].std() + 1e-6
        ref_mean, ref_std = ref[..., c].mean(), ref[..., c].std() + 1e-6
        out[..., c] = (img[..., c] - img_mean) / img_std * ref_std + ref_mean

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def class_swapped_color_stats(image, target_class_mean_std):
    """
    Alternative to hue_rotation — shift image color mean/std to match another
    class's precomputed statistics, rather than a single reference image.
    target_class_mean_std: (mean_rgb, std_rgb), each a length-3 sequence of
    floats in [0, 255] (compute these once, e.g. by averaging over all clean
    training images of the target class, before calling this).
    Only use if this is your chosen additional color intervention.
    """
    mean_rgb, std_rgb = target_class_mean_std
    img = np.array(image).astype(np.float32)

    out = np.empty_like(img)
    for c in range(3):
        img_mean, img_std = img[..., c].mean(), img[..., c].std() + 1e-6
        out[..., c] = (img[..., c] - img_mean) / img_std * std_rgb[c] + mean_rgb[c]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def translate(image, offset_px, direction, padding_mode="reflect"):
    """
    Translate by offset_px in {"up", "down", "left", "right"} using reflection
    padding followed by a shifted crop back to the original size.

    At offset_px = 0 this is an identity transform for every direction (used
    as the reference point on the translation curve).
    """
    if offset_px == 0:
        return image.copy()

    arr = np.array(image)
    H, W = arr.shape[:2]
    o = offset_px

    pad_kwargs = {"mode": padding_mode} if padding_mode != "reflect" else {"mode": "reflect"}
    padded = np.pad(arr, ((o, o), (o, o), (0, 0)), **pad_kwargs)

    # See derivation in README: cropping a (2*o)-shifted window of the
    # reflect-padded image is equivalent to shifting the visible content by
    # `o` pixels in the given direction, revealing reflected content on the
    # opposite edge.
    row_start = {"up": 2 * o, "down": 0, "left": o, "right": o}[direction]
    col_start = {"up": o, "down": o, "left": 2 * o, "right": 0}[direction]

    cropped = padded[row_start:row_start + H, col_start:col_start + W]
    return Image.fromarray(cropped)


def patch_shuffle(image, grid=(4, 4), seed=6304, image_index=None):
    """
    Divide into a grid[0] x grid[1] pixel-space grid, apply one non-identity
    permutation.

    NOTE on the assignment's "one non-identity patch permutation per image
    using seed 6304" wording — this is ambiguous between (a) a single global
    permutation reused for every image, or (b) a distinct-but-seeded
    permutation per image. Both satisfy "reuse exactly the same shuffled
    images across models" (the requirement that actually matters — models are
    compared on identical shuffled images). This function supports both:
      - image_index=None (default): ONE global permutation, same for every
        image passed through this function.
      - image_index=<int>: a distinct permutation per image, seeded
        deterministically as (seed + image_index) so it's still reproducible.
    Pick one interpretation and state your choice in the README/report.
    """
    arr = np.array(image)
    H, W = arr.shape[:2]
    gh, gw = grid
    assert H % gh == 0 and W % gw == 0, (
        f"Image size ({H}x{W}) must be divisible by grid {grid}."
    )
    ph, pw = H // gh, W // gw

    patches = [
        arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
        for i in range(gh) for j in range(gw)
    ]
    n = gh * gw

    effective_seed = seed if image_index is None else seed + image_index
    rng = random.Random(effective_seed)

    identity = list(range(n))
    perm = identity[:]
    while perm == identity:  # guarantee non-identity, per the assignment
        rng.shuffle(perm)

    shuffled = np.zeros_like(arr)
    idx = 0
    for i in range(gh):
        for j in range(gw):
            shuffled[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = patches[perm[idx]]
            idx += 1

    return Image.fromarray(shuffled)