# =============================================================================
# evaluate.py  –  Test-set evaluation: metrics, confusion matrix, GradCAM
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from config import DEVICE, CLASSES, MODELS_DIR, RESULTS_DIR, MODEL_NAMES
from dataset import get_dataloaders
from models import get_model
from utils import save_json


# ── Load best checkpoint ──────────────────────────────────────────────────────

def load_best_model(model_name: str) -> torch.nn.Module:
    ckpt = os.path.join(MODELS_DIR, f"best_{model_name}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. "
            f"Run train.py first."
        )
    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


# ── Collect predictions ───────────────────────────────────────────────────────

def get_predictions(model, loader):
    """Run model on a DataLoader; return labels, preds, and softmax probs."""
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred, average="weighted",
                                    zero_division=0), 4),
        "f1_macro" : round(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0), 4),
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=12)
    ax.set_yticklabels(CLASSES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True",      fontsize=13)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results_dict: dict, save_path: str):
    """
    Overlay ROC curves for all models (pothole = positive class, idx 1).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for (name, res), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_true"],
                                 res["y_probs"][:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves → {save_path}")


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Generates a heatmap highlighting the image regions most influential
    to the model's prediction, providing visual explainability.
    """

    def __init__(self, model: torch.nn.Module, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0,1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def _get_gradcam_layer(model_name: str, model):
    """Returns the last conv layer for each architecture."""
    if model_name == "custom_cnn":
        return model.features[-1][-3]          # last Conv2d in last block
    elif model_name == "vgg16":
        return model.features[-1]               # Conv2d(512,512)
    elif model_name == "mobilenetv2":
        return model.features[-1][0]            # Conv2d in last InvertedResidual
    raise ValueError(f"Unknown model: {model_name}")


def visualise_gradcam(model, model_name: str, loader, save_path: str,
                      n_samples: int = 6):
    """
    Saves a figure with n_samples images overlaid with GradCAM heatmaps.
    """
    target_layer = _get_gradcam_layer(model_name, model)
    gradcam      = GradCAM(model, target_layer)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    images_collected, labels_collected = [], []
    for imgs, lbls in loader:
        images_collected.append(imgs)
        labels_collected.append(lbls)
        if sum(len(b) for b in images_collected) >= n_samples:
            break

    images_tensor = torch.cat(images_collected)[:n_samples]
    labels_tensor = torch.cat(labels_collected)[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    fig.suptitle(f"Grad-CAM Explainability – {model_name}", fontsize=14)

    for i in range(n_samples):
        inp = images_tensor[i].unsqueeze(0).to(DEVICE).requires_grad_(True)
        cam, pred_idx = gradcam.generate(inp)

        # De-normalise for display
        img_np = images_tensor[i].cpu()
        img_np = img_np * std[:, None, None] + mean[:, None, None]
        img_np = img_np.permute(1, 2, 0).clamp(0, 1).numpy()

        # Resize cam to image size
        import cv2
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap     = cm.jet(cam_resized)[:, :, :3]
        overlay     = 0.5 * img_np + 0.5 * heatmap

        true_lbl = CLASSES[labels_tensor[i].item()]
        pred_lbl = CLASSES[pred_idx]
        colour   = "green" if true_lbl == pred_lbl else "red"

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {true_lbl}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(overlay, 0, 1))
        axes[1, i].set_title(f"Pred: {pred_lbl}", fontsize=9, color=colour)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Grad-CAM → {save_path}")


# ── Evaluate all models ───────────────────────────────────────────────────────

def evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    loaders, _ = get_dataloaders()
    test_loader = loaders["test"]

    all_results  = {}
    metrics_table = {}

    for name in MODEL_NAMES:
        print(f"\n  Evaluating: {name.upper()}")
        model  = load_best_model(name)
        y_true, y_pred, y_probs = get_predictions(model, test_loader)

        metrics = compute_metrics(y_true, y_pred)
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  F1 Score : {metrics['f1']*100:.2f}%")
        print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")

        plot_confusion_matrix(
            y_true, y_pred, name,
            os.path.join(RESULTS_DIR, f"cm_{name}.png")
        )

        try:
            visualise_gradcam(
                model, name, test_loader,
                os.path.join(RESULTS_DIR, f"gradcam_{name}.png")
            )
        except Exception as e:
            print(f"  GradCAM skipped ({e})")

        all_results[name]  = {"y_true": y_true, "y_probs": y_probs}
        metrics_table[name] = metrics

    # ── ROC curves ────────────────────────────────────────────────────────────
    plot_roc_curves(all_results,
                    os.path.join(RESULTS_DIR, "roc_curves.png"))

    # ── Comparison bar chart ──────────────────────────────────────────────────
    _plot_metric_comparison(metrics_table,
                            os.path.join(RESULTS_DIR, "metrics_comparison.png"))

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    save_json(metrics_table, os.path.join(RESULTS_DIR, "test_metrics.json"))
    print("\n  All evaluation artefacts saved to:", RESULTS_DIR)
    return metrics_table


def _plot_metric_comparison(metrics_table: dict, save_path: str):
    """Bar chart comparing Accuracy / Precision / Recall / F1 across models."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    x      = np.arange(len(metric_keys))
    width  = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, metrics) in enumerate(metrics_table.items()):
        vals = [metrics[k] * 100 for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i],
               alpha=0.85, edgecolor="white")

    ax.set_xlabel("Metric",  fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Test Set Performance – All Models", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metric_keys], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(50, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison chart → {save_path}")


if __name__ == "__main__":
    evaluate_all()
