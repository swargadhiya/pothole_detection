# =============================================================================
# train.py  –  Training pipeline with early stopping & LR scheduling
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE, SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, LR_FACTOR, LR_PATIENCE,
    RESULTS_DIR, MODELS_DIR, MODEL_NAMES,
)
from dataset import get_dataloaders
from models import get_model, count_parameters
from utils import plot_training_curves, save_json


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.
    Also saves the best checkpoint automatically.
    """
    def __init__(self, patience: int = PATIENCE, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_acc   = 0.0
        self.stop       = False

    def __call__(self, val_loss: float, val_acc: float,
                 model: nn.Module, path: str):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_acc  = val_acc
            torch.save(model.state_dict(), path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── Single epoch helpers ──────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, phase):
    """Run one epoch of training or validation."""
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                # Gradient clipping prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── Main training function ────────────────────────────────────────────────────

def train_model(model_name: str):
    """
    Full training loop for one model.

    Returns
    -------
    history : dict with lists of train_loss, train_acc, val_loss, val_acc
    best_acc : float, best validation accuracy achieved
    """
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders, _ = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(model_name).to(DEVICE)
    params = count_parameters(model)
    print(f"  Parameters – total: {params['total']:,}  |  "
          f"trainable: {params['trainable']:,}")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    ckpt_path = os.path.join(MODELS_DIR, f"best_{model_name}.pth")
    stopper   = EarlyStopping(patience=PATIENCE)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc = _run_epoch(model, loaders["train"], criterion,
                                   optimizer, "train")
        v_loss, v_acc = _run_epoch(model, loaders["valid"], criterion,
                                   optimizer, "valid")

        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train loss: {t_loss:.4f}  acc: {t_acc:.4f}  |  "
              f"Val loss: {v_loss:.4f}  acc: {v_acc:.4f}  |  "
              f"LR: {lr_now:.2e}")

        stopper(v_loss, v_acc, model, ckpt_path)
        if stopper.stop:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val accuracy: {stopper.best_acc:.4f}")
    print(f"  Checkpoint saved : {ckpt_path}")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(RESULTS_DIR, f"history_{model_name}.json")
    save_json(history, hist_path)

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, model_name,
                         os.path.join(RESULTS_DIR, f"curves_{model_name}.png"))

    return history, stopper.best_acc


# ── Train all models ──────────────────────────────────────────────────────────

def train_all():
    """Train every model defined in MODEL_NAMES and print a summary."""
    summary = {}
    for name in MODEL_NAMES:
        _, best_acc = train_model(name)
        summary[name] = round(best_acc * 100, 2)

    print("\n" + "="*40)
    print("  SUMMARY – Best Validation Accuracy")
    print("="*40)
    for name, acc in summary.items():
        print(f"  {name:<15}: {acc:.2f}%")
    print("="*40)

    save_json(summary, os.path.join(RESULTS_DIR, "training_summary.json"))
    return summary


if __name__ == "__main__":
    train_all()
