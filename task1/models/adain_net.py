"""
Vendored from naoto0804/pytorch-AdaIN (MIT License):
    https://github.com/naoto0804/pytorch-AdaIN
Unofficial PyTorch implementation of Huang & Belongie, "Arbitrary Style
Transfer in Real-time with Adaptive Instance Normalization" (ICCV 2017).

The `vgg` and `decoder` nn.Sequential definitions and the AdaIN math
(`calc_mean_std`, `adaptive_instance_normalization`) below are copied
VERBATIM from that repo's net.py / function.py, so that the pretrained
decoder.pth / vgg_normalised.pth weights (also from that repo's GitHub
release) load via load_state_dict() without key mismatches. `style_transfer`
is adapted from that repo's test.py (the multi-style interpolation path and
CLI wrapper are dropped since this assignment only needs single content/style
pairs). The training-only `Net` class, losses, and `coral` color-preservation
utility from the original repo are intentionally omitted — not needed for
generating cue-conflict images.

Attribute this file's origin in your README per the assignment's
"clearly attribute materially reused external code" requirement.
"""
import torch
import torch.nn as nn

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used (encoder truncates here)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_transfer(vgg_encoder, decoder_net, content, style, alpha=1.0):
    """
    Adapted from the original repo's test.py::style_transfer(). content and
    style: (N, 3, H, W) float tensors in [0, 1] (NOT ImageNet-normalized —
    vgg_normalised.pth already bakes its own fixed input transform into the
    first 1x1 conv layer, so feed it raw [0,1] RGB, not backbone-normalized
    tensors like the other three backbones expect).
    """
    assert 0.0 <= alpha <= 1.0
    content_f = vgg_encoder(content)
    style_f = vgg_encoder(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder_net(feat)


def load_adain_model(vgg_weights_path, decoder_weights_path, device="cpu"):
    """
    Loads the pretrained weights and truncates `vgg` to the first 31 layers
    (through relu4_1) as the original repo's test.py does — the full `vgg`
    Sequential above includes relu4_2 through relu5_4, which the decoder was
    never trained to invert.
    Returns (vgg_encoder, decoder_net), both frozen and in eval mode.
    """
    vgg_full = vgg
    decoder_net = decoder

    vgg_full.load_state_dict(torch.load(vgg_weights_path, map_location=device))
    decoder_net.load_state_dict(torch.load(decoder_weights_path, map_location=device))

    vgg_encoder = nn.Sequential(*list(vgg_full.children())[:31])

    vgg_encoder.eval().to(device)
    decoder_net.eval().to(device)

    for p in vgg_encoder.parameters():
        p.requires_grad = False
    for p in decoder_net.parameters():
        p.requires_grad = False

    return vgg_encoder, decoder_net