import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from shared.pacs import load_target_domain, PACS_CLASSES
from shared.pacs_protocol import make_eval_loader
from task2.models.backbone import build_resnet18
from task2.evaluation.class_analysis import per_class_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

target_folder = load_target_domain(cfg["data"]["root"])
target_loader = make_eval_loader(target_folder)

model = build_resnet18(n_classes=cfg["data"]["n_classes"], weights=None)
state = torch.load("task2/results/checkpoints/source_only.pt", map_location=device)
model.load_state_dict(state, strict=False)
model.eval().to(device)

accs, preds, labels = per_class_accuracy(model, target_loader, device, cfg["data"]["n_classes"])

print("\nSource-only per-class accuracy on Sketch (target):")
for name, acc in zip(PACS_CLASSES, accs):
    print(f"  {name:10s}: {acc:.4f}")
