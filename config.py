# =============================================================================
# config.py  –  Central configuration for all experiments
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update DATA_ROOT to wherever your Dataset/Dataset folder lives on your machine
DATA_ROOT   = os.path.join("Dataset")
TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
VALID_DIR   = os.path.join(DATA_ROOT, "valid")
TEST_DIR    = os.path.join(DATA_ROOT, "test")

RESULTS_DIR = "results"          # plots, confusion matrices, etc.
MODELS_DIR  = "saved_models"     # .pth checkpoints

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES     = ["normal", "pothole"]   # alphabetical = PyTorch ImageFolder default
NUM_CLASSES = 2

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # resize both H and W to this value
CHANNELS    = 3

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 10       # early-stopping patience (epochs without val-loss improvement)
LR_FACTOR       = 0.5      # ReduceLROnPlateau factor
LR_PATIENCE     = 5        # ReduceLROnPlateau patience

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Device (auto-detects GPU) ─────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model names (used as keys throughout the project) ─────────────────────────
MODEL_NAMES = ["custom_cnn", "vgg16", "mobilenetv2"]
