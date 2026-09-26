import json
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import open_clip
import matplotlib.pyplot as plt
import random


# ============================================================
# Config
# ============================================================

METADATA_PATH = "results/cue_conflict_metadata.json"
IMAGE_ROOT = "."

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

# Use MPS on Mac if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# Load metadata
# ============================================================

with open(METADATA_PATH) as f:
    metadata = json.load(f)


# ============================================================
# Load ResNet-50
# ============================================================

print("Loading ResNet-50...")

resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Replace ImageNet 1000-class head with 10-class head
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)

# Load trained CIFAR/STL-10 linear head
resnet_head = torch.load(
    "results/heads/resnet50_head.pt",
    map_location=DEVICE,
    weights_only=True,
)

resnet.fc.load_state_dict(resnet_head)

resnet.eval().to(DEVICE)


# ============================================================
# Load ViT-B/16
# ============================================================

print("Loading ViT-B/16...")

vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# ViT-B/16 classifier input dimension = 768
vit.heads.head = torch.nn.Linear(768, 10)

# Load trained 10-class head
vit_head = torch.load(
    "results/heads/vit_b_16_head.pt",
    map_location=DEVICE,
    weights_only=True,
)

# Handle either a state_dict or a complete module
if isinstance(vit_head, dict):
    vit.heads.head.load_state_dict(vit_head)
else:
    vit.heads.head = vit_head

vit.eval().to(DEVICE)


# ============================================================
# Load CLIP ViT-B/32
# ============================================================

print("Loading CLIP...")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
)

# OpenAI CLIP ViT-B/32 has 512-dimensional image features
clip_head = torch.nn.Linear(512, 10)

clip_head_state = torch.load(
    "results/heads/clip_vit_b_32_head.pt",
    map_location=DEVICE,
    weights_only=True,
)

# Handle either a state_dict or a complete module
if isinstance(clip_head_state, dict):
    clip_head.load_state_dict(clip_head_state)
else:
    clip_head = clip_head_state

clip_model.eval().to(DEVICE)
clip_head.eval().to(DEVICE)


# ============================================================
# Standard preprocessing
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Prediction
# ============================================================

def predict(model, img_tensor, model_type):
    with torch.no_grad():

        if model_type == "resnet":
            logits = model(img_tensor)

        elif model_type == "vit":
            logits = model(img_tensor)

        elif model_type == "clip":
            feat = clip_model.encode_image(img_tensor)

            # Normalize CLIP representation
            feat = feat / feat.norm(dim=-1, keepdim=True)

            logits = clip_head(feat)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        pred_idx = logits.argmax(dim=-1).item()

    return STL10_CLASSES[pred_idx]


# ============================================================
# Classify prediction as shape / texture / other
# ============================================================

def classify_prediction(pred, content_name, style_name):

    if pred == content_name:
        return "shape"

    elif pred == style_name:
        return "texture"

    else:
        return "other"


# ============================================================
# Select diverse examples
# ============================================================

random.seed(6304)
random.shuffle(metadata)

selected = []
seen_patterns = set()

for entry in metadata:

    img_path = f"{IMAGE_ROOT}/{entry['path']}"

    img = Image.open(img_path).convert("RGB")

    # ResNet / ViT preprocessing
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    # CLIP preprocessing
    clip_img_t = clip_preprocess(img).unsqueeze(0).to(DEVICE)

    # Predictions
    pred_resnet = predict(
        resnet,
        img_t,
        "resnet",
    )

    pred_vit = predict(
        vit,
        img_t,
        "vit",
    )

    pred_clip = predict(
        clip_model,
        clip_img_t,
        "clip",
    )

    # Shape / texture / other
    cat_resnet = classify_prediction(
        pred_resnet,
        entry["content_class_name"],
        entry["style_class_name"],
    )

    cat_vit = classify_prediction(
        pred_vit,
        entry["content_class_name"],
        entry["style_class_name"],
    )

    cat_clip = classify_prediction(
        pred_clip,
        entry["content_class_name"],
        entry["style_class_name"],
    )

    # Agreement pattern
    pattern = (
        cat_resnet,
        cat_vit,
        cat_clip,
    )

    entry["_preds"] = {
        "resnet50": (
            pred_resnet,
            cat_resnet,
        ),
        "vit_b_16": (
            pred_vit,
            cat_vit,
        ),
        "clip": (
            pred_clip,
            cat_clip,
        ),
    }

    # Keep only novel patterns
    if pattern not in seen_patterns:

        seen_patterns.add(pattern)
        selected.append(entry)

    if len(selected) >= 5:
        break


# ============================================================
# Plot selected examples
# ============================================================

n = len(selected)

if n == 0:
    raise RuntimeError(
        "No cue-conflict examples were selected. "
        "Check results/cue_conflict_metadata.json and image paths."
    )

fig, axes = plt.subplots(
    1,
    n,
    figsize=(4 * n, 5),
)

if n == 1:
    axes = [axes]


for ax, entry in zip(axes, selected):

    img = Image.open(
        f"{IMAGE_ROOT}/{entry['path']}"
    ).convert("RGB")

    ax.imshow(img)
    ax.axis("off")

    lines = [
        f"Content: {entry['content_class_name']} / "
        f"Style: {entry['style_class_name']}"
    ]

    for key, label in [
        ("resnet50", "RN50"),
        ("vit_b_16", "ViT"),
        ("clip", "CLIP"),
    ]:

        pred, cat = entry["_preds"][key]

        lines.append(
            f"{label}: {pred} ({cat})"
        )

    ax.set_title(
        "\n".join(lines),
        fontsize=9,
    )


plt.tight_layout()

plt.savefig(
    "cue_conflict_examples.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Print selected examples
# ============================================================

print("\nSelected examples:")

for entry in selected:

    print(
        entry["path"],
        entry["_preds"],
    )