# Code Generated for graph creation using Chat-GPT

import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

LOG_ROOT = "Phase2/Logs"

# Global Pose Loss
# Explicit model labels + colors (EDIT THESE)
MODEL_STYLES = {
    "VO": {"label": "VO",   "color": "tab:red"},
    "VIO": {"label": "VIO",   "color": "tab:green"},
    "IO": {"label": "IO",   "color": "tab:blue"},
}

# TensorBoard scalar tags (EDIT IF NEEDED)
TAGS = {
    "GlobalTrainLoss": "Pose_epoch_train_loss",
    "GlobalValLoss":   "Pose_epoch_val_loss",
    "TwistTrainLoss": "Twist_epoch_train_loss",
    "TwistValLoss": "Twist_epoch_val_loss",
}

# ----------------------------------------

def load_scalar(run_dir, tag):
    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise KeyError(f"Tag '{tag}' not found in {run_dir}")

    events = ea.Scalars(tag)

    # 1 datapoint = 1 epoch
    epochs = list(range(1, len(events) + 1))
    values = [e.value for e in events]

    return epochs, values


# Load all data
data = {}

for model_name in MODEL_STYLES:
    run_path = os.path.join(LOG_ROOT, model_name)
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"Missing run folder: {run_path}")

    data[model_name] = {}
    for key, tag in TAGS.items():
        data[model_name][key] = load_scalar(run_path, tag)


def plot_metric(metric_key, title, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for model_name, style in MODEL_STYLES.items():
        epochs, values = data[model_name][metric_key]

        # mult = 100000 / (len(epochs))

        # epochs_scaled = []
        # for i in epochs:
        #     x = (i)*mult
        #     epochs_scaled.append(x)

        plt.plot(
            epochs,
            values,
            label=style["label"],
            color=style["color"],
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Phase2/Output/" + filename, dpi=300)
    plt.show()


# -------- Generate the 4 plots --------

plot_metric(
    "GlobalTrainLoss",
    "Global Pose Training Loss vs Epochs",
    "Loss",
    "glob_train_loss_epochs.png"
)

plot_metric(
    "GlobalValLoss",
    "Global Pose Val Loss vs Epochs",
    "Loss",
    "glob_val_loss_epochs.png"
)

plot_metric(
    "TwistTrainLoss",
    "Twist Train Loss vs Epochs",
    "Loss",
    "twist_train_loss_epochs.png"
)

plot_metric(
    "TwistValLoss",
    "Twist Val Loss vs Epochs",
    "Loss",
    "twist_val_loss_epochs.png"
)