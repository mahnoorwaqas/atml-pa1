"""
Downloads the pretrained AdaIN encoder/decoder weights from the original
repo's GitHub release (naoto0804/pytorch-AdaIN, MIT License):
    https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0
These URLs were verified working directly (decoder.pth ~14MB, vgg_normalised.pth ~80MB).

Usage:
    python task1/data/download_adain_weights.py --out_dir task1/models/adain_weights
"""
import argparse
import os
import urllib.request

_BASE_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0"
_FILES = ["decoder.pth", "vgg_normalised.pth"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="task1/models/adain_weights")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in _FILES:
        dest = os.path.join(args.out_dir, fname)
        if os.path.exists(dest):
            print(f"Already have {dest}, skipping.")
            continue
        url = f"{_BASE_URL}/{fname}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {os.path.getsize(dest)} bytes")


if __name__ == "__main__":
    main()