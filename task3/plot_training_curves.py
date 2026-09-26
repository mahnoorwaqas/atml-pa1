import json
from pathlib import Path
import matplotlib.pyplot as plt


RESULTS = Path("task3/results")


def load_log(path):
    with open(path, "r") as f:
        data = json.load(f)

    history = data.get("history", data)
    return history


def epoch_average(history, key):
    values = {}

    for item in history:
        if key not in item:
            continue

        epoch = item.get("epoch")
        if epoch is None:
            continue

        values.setdefault(epoch, []).append(float(item[key]))

    epochs = sorted(values)
    means = [sum(values[e]) / len(values[e]) for e in epochs]

    return epochs, means


# =========================================================
# Training logs
# =========================================================

classification_logs = {
    "ERM": (
        Path("task2/results/checkpoints/source_only_training_log.json"),
        "cls_loss",
    ),
    "DAN-DG": (
        RESULTS / "checkpoints" / "dan_dg_training_log.json",
        "cls_loss",
    ),
    "SAM": (
        RESULTS / "checkpoints" / "sam_training_log.json",
        "loss_at_original",
    ),
}


# =========================================================
# Classification loss: ERM vs DAN-DG vs SAM
# =========================================================

plt.figure(figsize=(8, 5))

for name, (path, loss_key) in classification_logs.items():

    if not path.exists():
        print(f"WARNING: {path} not found")
        continue

    history = load_log(path)

    epochs, loss = epoch_average(history, loss_key)

    if not loss:
        print(f"WARNING: no {loss_key} found in {path}")
        continue

    plt.plot(
        epochs,
        loss,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Classification Loss")
plt.title("Task 3 Classification Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "task3_classification_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    RESULTS / "task3_classification_loss.png"
)


# =========================================================
# DAN-DG MMD loss
# =========================================================

dan_path = RESULTS / "checkpoints" / "dan_dg_training_log.json"

if dan_path.exists():

    history = load_log(dan_path)

    epochs, mmd = epoch_average(
        history,
        "mmd_loss"
    )

    if mmd:

        plt.figure(figsize=(8, 5))

        plt.plot(
            epochs,
            mmd,
            label="DAN-DG MMD"
        )

        plt.xlabel("Epoch")
        plt.ylabel("MMD Loss")
        plt.title("Task 3 DAN-DG MMD Penalty")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            RESULTS / "task3_mmd_loss.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Saved:",
            RESULTS / "task3_mmd_loss.png"
        )

    else:
        print(
            "WARNING: no mmd_loss found in DAN-DG log"
        )

else:
    print(
        f"WARNING: {dan_path} not found"
    )