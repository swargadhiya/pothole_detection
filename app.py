# =============================================================================
# app.py  –  Gradio web interface (Hugging Face Spaces deployment)
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
# =============================================================================
#
# Deploy to Hugging Face Spaces:
#   1. Create a new Space (SDK = Gradio)
#   2. Upload app.py, predict.py, models.py, config.py, utils.py,
#      requirements.txt, and your saved_models/ folder
#   3. The Space will auto-install requirements and launch this file
# =============================================================================

import os
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torch
import cv2

from predict import PotholeDetector, _EVAL_TRANSFORM
from config import CLASSES, DEVICE, MODELS_DIR
from models import get_model
from evaluate import GradCAM, _get_gradcam_layer


# ── Load all three detectors once at startup ──────────────────────────────────

_detectors = {}

def _load_detectors():
    for name in ("custom_cnn", "vgg16", "mobilenetv2"):
        ckpt = os.path.join(MODELS_DIR, f"best_{name}.pth")
        if os.path.exists(ckpt):
            _detectors[name] = PotholeDetector(name)
    if not _detectors:
        raise RuntimeError(
            "No model checkpoints found in saved_models/. "
            "Run train.py first."
        )

_load_detectors()


# ── GradCAM overlay helper ────────────────────────────────────────────────────

def _make_gradcam_overlay(pil_image: Image.Image, model_name: str) -> Image.Image:
    """Returns an RGB PIL image of the GradCAM overlay."""
    model  = _detectors[model_name].model
    target = _get_gradcam_layer(model_name, model)
    gc     = GradCAM(model, target)

    tensor    = _EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    cam, pred = gc.generate(tensor)

    img_np = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    cam_rs = cv2.resize(cam, (224, 224))
    heat   = mpl_cm.jet(cam_rs)[:, :, :3]
    blend  = np.clip(0.55 * img_np + 0.45 * heat, 0, 1)
    return Image.fromarray((blend * 255).astype(np.uint8))


# ── Prediction function ───────────────────────────────────────────────────────

def predict(image: np.ndarray, model_choice: str):
    """
    Called by Gradio on every submission.

    Parameters
    ----------
    image        : numpy array from the Gradio Image component
    model_choice : display name selected in the radio button

    Returns
    -------
    verdict_text : str  – formatted prediction + confidence
    bar_chart    : PIL  – horizontal confidence bar chart
    gradcam_img  : PIL  – GradCAM heatmap overlay
    """
    name_map = {
        "Custom CNN (from scratch)": "custom_cnn",
        "VGG16 (transfer learning)": "vgg16",
        "MobileNetV2 (lightweight)" : "mobilenetv2",
    }
    model_name = name_map[model_choice]

    if model_name not in _detectors:
        return f"⚠️ Model '{model_name}' not loaded.", None, None

    pil_img              = Image.fromarray(image.astype("uint8")).convert("RGB")
    label, conf, probs   = _detectors[model_name].predict(pil_img)

    # ── Verdict text ─────────────────────────────────────────────────────────
    emoji   = "🚨 POTHOLE DETECTED" if label == "pothole" else "✅ ROAD IS NORMAL"
    verdict = (
        f"{emoji}\n\n"
        f"Model       : {model_choice}\n"
        f"Prediction  : {label.capitalize()}\n"
        f"Confidence  : {conf*100:.1f}%\n"
        f"Normal prob : {probs['normal']*100:.1f}%\n"
        f"Pothole prob: {probs['pothole']*100:.1f}%"
    )

    # ── Confidence bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 2))
    colors  = ["#2ecc71", "#e74c3c"]
    bars    = ax.barh(CLASSES,
                      [probs["normal"] * 100, probs["pothole"] * 100],
                      color=colors, height=0.5)
    for bar, val in zip(bars, [probs["normal"], probs["pothole"]]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Class Probabilities", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.canvas.draw()
    chart_img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    try:
        gradcam_img = _make_gradcam_overlay(pil_img, model_name)
    except Exception as e:
        gradcam_img = pil_img          # fallback: show original image
        print(f"GradCAM error: {e}")

    return verdict, chart_img, gradcam_img


# ── Example images (add a few to the repo for demo) ──────────────────────────

examples = [
    ["examples/pothole_1.jpg", "MobileNetV2 (lightweight)"],
    ["examples/normal_1.jpg",  "MobileNetV2 (lightweight)"],
]
# Filter to only paths that actually exist (avoids Gradio errors)
examples = [e for e in examples if os.path.exists(e[0])]


# ── Gradio interface ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="🚗 Pothole Detection",
    theme=gr.themes.Soft(),
    css="""
        .container { max-width: 900px; margin: auto; }
        .verdict   { font-size: 1.1rem; font-family: monospace; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🚗 Pothole Detection System
        **Module:** CMP-L016 Deep Learning Applications  
        **Student:** Swar Anilbhai Gadhiya (A00075099)  
        **MSc Data Science | University of Roehampton**

        Upload a road image and select a model to detect potholes.  
        The system uses three deep learning architectures trained on 
        the Kaggle Pothole Detection dataset.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Road Image",
                type="numpy",
            )
            model_radio = gr.Radio(
                choices=[
                    "Custom CNN (from scratch)",
                    "VGG16 (transfer learning)",
                    "MobileNetV2 (lightweight)",
                ],
                value="MobileNetV2 (lightweight)",
                label="🧠 Select Model",
            )
            submit_btn = gr.Button("🔍 Analyse", variant="primary")

        with gr.Column(scale=1):
            verdict_box  = gr.Textbox(
                label="📊 Prediction Result",
                lines=8,
                elem_classes=["verdict"],
            )
            chart_output = gr.Image(label="📈 Confidence Chart")

    with gr.Row():
        gradcam_output = gr.Image(
            label="🔥 Grad-CAM Explainability "
                  "(highlights regions that influenced the prediction)"
        )

    submit_btn.click(
        fn=predict,
        inputs=[input_image, model_radio],
        outputs=[verdict_box, chart_output, gradcam_output],
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=[input_image, model_radio],
            label="📁 Example Images",
        )

    gr.Markdown(
        """
        ---
        **How it works:**  
        1. Upload any road/street-level image.  
        2. Choose a model (MobileNetV2 is fastest; VGG16 is most accurate).  
        3. Click Analyse — the system predicts Normal or Pothole with confidence.  
        4. Grad-CAM shows *where* in the image the model focused its attention.

        **Models trained on:** Kaggle Pothole Detection Dataset  
        (800 train / 100 valid / 108 test images, binary classification)
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)   # share=True gives a public link for demo
