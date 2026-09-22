import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ImageNet normalization (ResNet-50 and ViT-B/16 pretrained weights both use this)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI CLIP normalization (used by the 'openai' pretrained OpenCLIP checkpoints)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize(batch, mean, std, device):
    """Per-channel normalization only — no resize/crop (that already happened
    upstream in the shared 224x224 intervention pipeline)."""
    batch = batch.to(device)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (batch - mean_t) / std_t


class ResNet50Backbone:
    """Global-average-pooled feature from torchvision ResNet-50, IMAGENET1K_V2 weights."""

    def __init__(self, weights="IMAGENET1K_V2", device=None):
        from torchvision.models import resnet50, ResNet50_Weights

        self.device = device or _default_device()
        weights_enum = getattr(ResNet50_Weights, weights) if weights is not None else None
        self.model = resnet50(weights=weights_enum)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.feature_dim = 2048

    def preprocess(self, batch):
        return _normalize(batch, _IMAGENET_MEAN, _IMAGENET_STD, self.device)

    def extract_features(self, batch):
        x = self.preprocess(batch)
        with torch.no_grad():
            return self.model(x)


class ViTB16Backbone:
    """Final class token from torchvision ViT-B/16, IMAGENET1K_V1 weights."""

    def __init__(self, weights="IMAGENET1K_V1", device=None):
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        self.device = device or _default_device()
        weights_enum = getattr(ViT_B_16_Weights, weights) if weights is not None else None
        self.model = vit_b_16(weights=weights_enum)
        self.model.heads = nn.Identity()
        self.model.eval().to(self.device)
        self.feature_dim = 768

    def preprocess(self, batch):
        return _normalize(batch, _IMAGENET_MEAN, _IMAGENET_STD, self.device)

    def extract_features(self, batch):
        x = self.preprocess(batch)
        with torch.no_grad():
            return self.model(x)


class CLIPViTB32Backbone:
    """Normalized image embedding from OpenCLIP ViT-B-32, pretrained='openai'."""

    def __init__(self, pretrained="openai", device=None):
        import open_clip

        self.device = device or _default_device()
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval().to(self.device)
        self.feature_dim = 512

    def preprocess(self, batch):
        return _normalize(batch, _CLIP_MEAN, _CLIP_STD, self.device)

    def extract_features(self, batch):
        """L2-normalized CLIP image embedding, per the assignment's spec."""
        x = self.preprocess(batch)
        with torch.no_grad():
            features = self.model.encode_image(x)
            return features / features.norm(dim=-1, keepdim=True)

    def zero_shot_logits(self, batch, class_names, prompt_template):
        """
        Cosine similarity between image embeddings and text embeddings of
        prompt_template.format(class=c) for each class c, scaled by CLIP's
        logit_scale. Softmax this externally (e.g. in evaluate_bias.py) to get
        the "confidence from the softmax over scaled class similarities" the
        assignment asks for.
        """
        image_features = self.extract_features(batch)

        prompts = [prompt_template.format(**{"class": c}) for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            similarities = logit_scale * image_features @ text_features.T

        return similarities


class LinearHead:
    """Trainable linear classifier on top of a frozen backbone's features."""

    def __init__(self, in_dim, n_classes):
        self.model = nn.Linear(in_dim, n_classes)

    def fit(self, train_features, train_labels, val_features, val_labels, cfg):
        """
        AdamW, lr=1e-3, weight_decay=1e-4, max 50 epochs, early stop after 5
        epochs without val-accuracy improvement, seed 6304.

        Trains full-batch (not mini-batched) each epoch — with a handful of
        thousand pre-extracted feature vectors this fits comfortably in
        memory, so there's no need for a DataLoader here.
        """
        device = train_features.device
        self.model.to(device)

        cfg_head = cfg["classifier_head"]
        torch.manual_seed(cfg["seed"])

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg_head["lr"],
            weight_decay=cfg_head["weight_decay"],
        )

        best_val_acc = -1.0
        best_state = None
        epochs_without_improvement = 0
        epoch = 0

        for epoch in tqdm(
            range(cfg_head["max_epochs"]),
            desc="Training linear head",
            unit="epoch",
        ):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(train_features)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_features)
                val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg_head["early_stop_patience"]:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_val_accuracy": best_val_acc, "epochs_trained": epoch + 1}

    def predict_logits(self, features):
        self.model.eval()
        with torch.no_grad():
            return self.model(features)


def get_backbone(name, cfg):
    """Factory: name in {'resnet50', 'vit_b_16', 'clip_vit_b_32'} -> backbone instance."""
    device = _default_device()
    backbone_cfg = cfg["backbones"]

    if name == "resnet50":
        return ResNet50Backbone(weights=backbone_cfg["resnet50"]["weights"], device=device)
    elif name == "vit_b_16":
        return ViTB16Backbone(weights=backbone_cfg["vit_b_16"]["weights"], device=device)
    elif name == "clip_vit_b_32":
        return CLIPViTB32Backbone(
            pretrained=backbone_cfg["clip_vit_b_32"]["pretrained"], device=device
        )
    else:
        raise ValueError(f"Unknown backbone '{name}'")