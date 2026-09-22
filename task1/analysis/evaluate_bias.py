"""
Evaluation metrics shared across clean baseline, color bias, cue-conflict,
translation, and patch-shuffle experiments.
"""
import numpy as np
from sklearn.metrics import f1_score


def _to_numpy(x):
    """Accept torch tensors or numpy arrays/lists; always return a numpy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_clean_metrics(logits, labels):
    """
    logits: (N, C) array of class scores (raw logits or softmax probabilities —
            both give the same argmax and max-confidence-after-softmax result
            as long as you pass softmax probabilities for 'confidence').
    labels: (N,) array of integer ground-truth class indices.

    Returns dict: {top1_accuracy, macro_f1, mean_max_confidence}.

    For CLIP zero-shot, pass softmax(scaled_similarities) as `logits` so that
    "confidence" here is genuinely a probability, per the assignment's wording
    ("compute confidence from the softmax over scaled class similarities").
    """
    logits = _to_numpy(logits)
    labels = _to_numpy(labels)

    preds = np.argmax(logits, axis=1)
    top1_accuracy = float(np.mean(preds == labels))
    macro_f1 = float(f1_score(labels, preds, average="macro"))

    # If `logits` are raw (unnormalized) logits rather than probabilities,
    # convert with softmax before taking "confidence" so this number is
    # meaningful as a probability in [0, 1].
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))  # numerically stable
    probs = exp / exp.sum(axis=1, keepdims=True)
    mean_max_confidence = float(np.mean(probs.max(axis=1)))

    return {
        "top1_accuracy": top1_accuracy,
        "macro_f1": macro_f1,
        "mean_max_confidence": mean_max_confidence,
    }


def compute_intervention_metrics(clean_logits, intervened_logits, labels):
    """
    clean_logits, intervened_logits: (N, C) arrays, same image order.
    labels: (N,) ground-truth class indices.

    Returns dict: {accuracy, accuracy_delta_vs_clean, prediction_consistency}.
    prediction_consistency = fraction of images whose predicted class is
    unchanged between clean and intervened predictions (this does NOT require
    the prediction to be correct — it's clean-pred == intervened-pred).
    """
    clean_logits = _to_numpy(clean_logits)
    intervened_logits = _to_numpy(intervened_logits)
    labels = _to_numpy(labels)

    clean_preds = np.argmax(clean_logits, axis=1)
    intervened_preds = np.argmax(intervened_logits, axis=1)

    accuracy = float(np.mean(intervened_preds == labels))
    clean_accuracy = float(np.mean(clean_preds == labels))
    accuracy_delta_vs_clean = accuracy - clean_accuracy

    prediction_consistency = float(np.mean(intervened_preds == clean_preds))

    return {
        "accuracy": accuracy,
        "accuracy_delta_vs_clean": accuracy_delta_vs_clean,
        "prediction_consistency": prediction_consistency,
    }


def classify_cue_conflict_predictions(predictions, content_labels, style_labels):
    """
    predictions, content_labels, style_labels: (N,) arrays of integer class
    indices (predictions = argmax of model logits on the cue-conflict images;
    content_labels = the shape/content class of each conflict image;
    style_labels = the texture/style class of each conflict image).

    For each image, classify the prediction as:
      "shape"   if predictions == content_labels
      "texture" if predictions == style_labels
      "other"   otherwise (neither intended class — e.g. a third class
                entirely, or a tie broken toward neither)

    Returns:
      classifications: (N,) array of dtype '<U7' with values in
                        {"shape", "texture", "other"}
      counts: dict {"n_shape": int, "n_texture": int, "n_other": int}
    """
    predictions = _to_numpy(predictions)
    content_labels = _to_numpy(content_labels)
    style_labels = _to_numpy(style_labels)

    is_shape = predictions == content_labels
    is_texture = (~is_shape) & (predictions == style_labels)
    is_other = ~(is_shape | is_texture)

    classifications = np.empty(predictions.shape[0], dtype="<U7")
    classifications[is_shape] = "shape"
    classifications[is_texture] = "texture"
    classifications[is_other] = "other"

    counts = {
        "n_shape": int(is_shape.sum()),
        "n_texture": int(is_texture.sum()),
        "n_other": int(is_other.sum()),
    }

    return classifications, counts


def shape_bias_and_coverage(n_shape, n_texture, n_total):
    """
    Shape Bias (%) = n_shape / (n_shape + n_texture) * 100
    Coverage (%)   = (n_shape + n_texture) / n_total * 100

    Returns dict: {shape_bias_pct, coverage_pct}.
    shape_bias_pct is None if n_shape + n_texture == 0 (undefined — flag this
    rather than silently returning 0, since 0% shape bias and "undefined" mean
    very different things when reporting on a model with near-zero coverage).
    """
    denom = n_shape + n_texture
    shape_bias_pct = (n_shape / denom * 100.0) if denom > 0 else None
    coverage_pct = (denom / n_total * 100.0) if n_total > 0 else 0.0

    return {
        "shape_bias_pct": shape_bias_pct,
        "coverage_pct": coverage_pct,
    }


def translation_curve(clean_logits, translated_logits_by_offset, labels):
    """
    clean_logits: (N, C) array.
    translated_logits_by_offset: dict mapping offset_px (e.g. 0, 8, 16, 32) to
        a list of (N, C) arrays, one per direction (e.g. [up, down, left, right]),
        all on the SAME N images in the SAME order as clean_logits.
    labels: (N,) ground-truth class indices.

    For each offset, averages accuracy and consistency across the provided
    directions (each direction scored independently against `labels` /
    `clean_logits`, then the per-direction numbers are averaged — this matches
    "Translate ... and average the results across directions").

    Returns: {offset_px: {"accuracy": float, "consistency": float}}
    """
    labels = _to_numpy(labels)
    clean_logits = _to_numpy(clean_logits)

    curve = {}
    for offset_px, per_direction_logits in translated_logits_by_offset.items():
        accuracies = []
        consistencies = []
        for direction_logits in per_direction_logits:
            metrics = compute_intervention_metrics(clean_logits, direction_logits, labels)
            accuracies.append(metrics["accuracy"])
            consistencies.append(metrics["prediction_consistency"])

        curve[offset_px] = {
            "accuracy": float(np.mean(accuracies)),
            "consistency": float(np.mean(consistencies)),
        }

    return curve