# =============================================================================
# predict.py  –  PotholeDetector class for single-image inference
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================

import os
from pathlib import Path
from typing import Union

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import DEVICE, CLASSES, MODELS_DIR, IMG_SIZE
from models import get_model


# ── Transform (same as eval) ──────────────────────────────────────────────────

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


# ── Detector class ────────────────────────────────────────────────────────────

class PotholeDetector:
    """
    High-level inference wrapper.

    Usage
    -----
    >>> detector = PotholeDetector("mobilenetv2")
    >>> label, confidence = detector.predict("road_image.jpg")
    >>> print(label, confidence)
    'pothole'  0.9832
    """

    def __init__(self, model_name: str = "mobilenetv2"):
        if model_name not in ("custom_cnn", "vgg16", "mobilenetv2"):
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.model      = get_model(model_name).to(DEVICE)

        ckpt = os.path.join(MODELS_DIR, f"best_{model_name}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Run train.py first."
            )

        self.model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE)
        )
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> tuple[str, float, dict]:
        """
        Predict whether a road image contains a pothole.

        Parameters
        ----------
        image_input : path string, PIL Image, or numpy array (H×W×C, uint8)

        Returns
        -------
        label       : str   – 'normal' or 'pothole'
        confidence  : float – probability of the predicted class  [0, 1]
        probs_dict  : dict  – {'normal': p0, 'pothole': p1}
        """
        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype("uint8")).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        # ── Pre-process ───────────────────────────────────────────────────────
        tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        label      = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        probs_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}

        return label, confidence, probs_dict

    def batch_predict(self, image_paths: list) -> list[dict]:
        """Predict on a list of image paths; returns list of result dicts."""
        results = []
        for path in image_paths:
            label, conf, probs = self.predict(path)
            results.append({
                "path":       str(path),
                "prediction": label,
                "confidence": conf,
                "probs":      probs,
            })
        return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_name]")
        print("       model_name: custom_cnn | vgg16 | mobilenetv2")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv2"

    detector = PotholeDetector(model_name)
    label, conf, probs = detector.predict(img_path)

    print(f"\n  Image      : {img_path}")
    print(f"  Model      : {model_name}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {conf*100:.1f}%")
    print(f"  Probs      : normal={probs['normal']*100:.1f}%  "
          f"pothole={probs['pothole']*100:.1f}%")
