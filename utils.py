# =============================================================================
# utils.py  –  Shared helper functions (plotting, JSON I/O)
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import json
import os
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, model_name: str, save_path: str):
    """
    Saves a 1×2 figure:
      left  – training & validation loss over epochs
      right – training & validation accuracy over epochs
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves – {model_name}", fontsize=14)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train",
             color="#e74c3c", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Validation",
             color="#3498db", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, label="Train",
             color="#e74c3c", lw=2)
    ax2.plot(epochs, val_acc_pct,   label="Validation",
             color="#3498db", lw=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
