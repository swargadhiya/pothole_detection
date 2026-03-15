# =============================================================================
# models.py  –  All three model architectures
# Pothole Detection | Swar Anilbhai Gadhiya | A00075099
#
#   1. CustomCNN       – built from scratch, road-specific feature learning
#   2. VGG16Model      – pre-trained VGG16, frozen base + custom head
#   3. MobileNetV2Model– pre-trained MobileNetV2, lightweight / edge-ready
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# 1.  C U S T O M   C N N
# ─────────────────────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """
    A 5-block convolutional network built entirely from scratch.

    Design rationale
    ────────────────
    • Each block doubles the number of feature maps: 32→64→128→256→512,
      following the pattern of VGG but with far fewer parameters.
    • BatchNorm after every Conv prevents internal covariate shift and
      stabilises training on our small dataset.
    • MaxPool halves spatial dimensions; adaptive average pool collapses
      the final map to 1×1 regardless of input size.
    • Dropout (0.5) in the classifier head combats overfitting.
    • Leaky-ReLU (negative slope 0.01) avoids dying-neuron problems.
    """

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3,   32),    # 224 → 112
            self._conv_block(32,  64),    # 112 → 56
            self._conv_block(64,  128),   # 56  → 28
            self._conv_block(128, 256),   # 28  → 14
            self._conv_block(256, 512),   # 14  → 7
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # 7 → 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for Conv layers; Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  V G G 1 6   T R A N S F E R   L E A R N I N G
# ─────────────────────────────────────────────────────────────────────────────

class VGG16Model(nn.Module):
    """
    VGG-16 with ImageNet-pretrained weights.

    Strategy
    ────────
    • Freeze all convolutional blocks (blocks 1–4) to preserve low-level
      ImageNet features (edges, textures).
    • Fine-tune block 5 (the last two Conv layers) so the network can
      adapt its high-level representations to road-surface textures.
    • Replace the original 4096→4096→1000 head with a lighter
      4096→256→num_classes head with BatchNorm + Dropout.

    This two-stage approach (freeze then fine-tune) is particularly effective
    when the target domain (road images) differs from ImageNet but shares
    low-level visual statistics.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze all feature layers first
        for param in backbone.features.parameters():
            param.requires_grad = False

        # Un-freeze block-5 (layers 24–30 in vgg16.features)
        for layer in backbone.features[24:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.features   = backbone.features
        self.avgpool    = backbone.avgpool      # adaptive 7×7

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  M O B I L E N E T V 2   T R A N S F E R   L E A R N I N G
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV2Model(nn.Module):
    """
    MobileNetV2 with ImageNet-pretrained weights.

    Strategy
    ────────
    • MobileNetV2 uses inverted residual blocks with depthwise separable
      convolutions, making it ~8× smaller than VGG16 in parameters.
    • Freeze the first 14 of 19 inverted-residual blocks to preserve
      general features; fine-tune the last 5 for domain adaptation.
    • Replace the default classifier with a BatchNorm + Dropout head.

    Chosen for edge/mobile deployment readiness as stated in Milestone 1.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Freeze early feature layers (features[0:14])
        for param in backbone.features[:14].parameters():
            param.requires_grad = False

        # Fine-tune layers 14–18 + conv_last
        for param in backbone.features[14:].parameters():
            param.requires_grad = True

        self.features = backbone.features           # outputs 1280 channels
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(name: str) -> nn.Module:
    """
    Returns an instantiated model by name.
    Valid names: 'custom_cnn', 'vgg16', 'mobilenetv2'
    """
    registry = {
        "custom_cnn" : CustomCNN,
        "vgg16"      : VGG16Model,
        "mobilenetv2": MobileNetV2Model,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name]()


def count_parameters(model: nn.Module) -> dict:
    """Returns total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
