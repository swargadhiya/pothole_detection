# =============================================================================
# dataset.py  –  Data loading, augmentation, and DataLoaders
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, SEED
)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms():
    """
    Returns a dict of torchvision transforms for each split.

    Training  – aggressive augmentation to improve generalisation:
        horizontal/vertical flip, rotation ±20°, colour jitter,
        random affine (shear + translate), random erasing.

    Valid/Test – only resize + normalise (no random ops).

    ImageNet mean/std are used because VGG16 and MobileNetV2 are
    pretrained on ImageNet; the Custom CNN benefits from the same
    normalisation for a fair comparison.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.2, hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),   # simulate occlusion
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {"train": train_transform, "valid": eval_transform, "test": eval_transform}


# ── Datasets ──────────────────────────────────────────────────────────────────

def get_datasets():
    """
    Returns a dict of ImageFolder datasets for train / valid / test splits.
    ImageFolder expects:  split_dir / class_name / image.jpg
    which matches our structure: train/normal/… and train/pothole/…
    """
    tfms = get_transforms()
    return {
        "train": datasets.ImageFolder(TRAIN_DIR, transform=tfms["train"]),
        "valid": datasets.ImageFolder(VALID_DIR, transform=tfms["valid"]),
        "test" : datasets.ImageFolder(TEST_DIR,  transform=tfms["test"]),
    }


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders():
    """
    Returns a dict of DataLoaders.
    num_workers=0 is safe on Windows; increase on Linux/macOS for speed.
    """
    g = torch.Generator()
    g.manual_seed(SEED)

    dsets = get_datasets()

    loaders = {
        "train": DataLoader(
            dsets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=g,
        ),
        "valid": DataLoader(
            dsets["valid"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test": DataLoader(
            dsets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }
    return loaders, dsets
