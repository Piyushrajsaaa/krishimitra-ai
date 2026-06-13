"""
KrishiMitra AI — EfficientNet-B0 Disease Classifier
Fine-tuned on PlantVillage dataset (38 disease classes).
Uses transfer learning for fast training on CPU.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger


# ── 38 PlantVillage Disease Classes ───────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── Human-readable display names ──────────────────────────
DISPLAY_NAMES = {cls: cls.replace("___", " — ").replace("_", " ")
                 for cls in DISEASE_CLASSES}


def build_model(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    Builds EfficientNet-B0 with custom classification head.
    Uses transfer learning — freezes base, trains only head.

    Args:
        num_classes: Number of disease classes (38 for PlantVillage)
        pretrained: Load ImageNet weights for transfer learning

    Returns:
        PyTorch model ready for training or inference
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all base layers (transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with our custom head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    logger.info(f"EfficientNet-B0 built | classes: {num_classes} | pretrained: {pretrained}")
    return model


def get_device() -> torch.device:
    """Returns best available device (CUDA > CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


class DiseaseClassifier:
    """
    Wrapper around EfficientNet-B0 for inference.
    Handles model loading, prediction, and confidence scoring.
    """

    def __init__(self):
        self.device = get_device()
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self.classes = DISEASE_CLASSES
        self.threshold = settings.CONFIDENCE_THRESHOLD

    def load(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Loads model from checkpoint or initializes fresh model.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.model = build_model(num_classes=len(self.classes))

            path = checkpoint_path or settings.MODEL_CHECKPOINT_PATH
            if Path(path).exists():
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"Model loaded from checkpoint: {path}")
            else:
                logger.warning(f"No checkpoint at {path} — using untrained model")

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    def predict(self, tensor: torch.Tensor) -> Dict:
        """
        Runs inference on preprocessed image tensor.

        Args:
            tensor: Shape [1, 3, 224, 224]

        Returns:
            Dict with disease_name, confidence, top5, escalate
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load() first.")
            return {}

        try:
            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)

            # Top prediction
            confidence, predicted_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

            disease_class = self.classes[predicted_idx]
            display_name = DISPLAY_NAMES[disease_class]

            # Top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
            top5 = [
                {
                    "disease": DISPLAY_NAMES[self.classes[idx]],
                    "confidence": round(prob, 4)
                }
                for idx, prob in zip(
                    top5_indices[0].tolist(),
                    top5_probs[0].tolist()
                )
            ]

            # Escalate if confidence below threshold
            escalate = confidence < self.threshold

            result = {
                "disease_class": disease_class,
                "disease_name": display_name,
                "confidence": round(confidence, 4),
                "top5": top5,
                "escalate_to_expert": escalate,
                "is_healthy": "healthy" in disease_class.lower(),
            }

            logger.info(
                f"Prediction: {display_name} | "
                f"confidence: {confidence:.2%} | "
                f"escalate: {escalate}"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}

    def get_crop_from_class(self, disease_class: str) -> str:
        """Extracts crop name from disease class string."""
        return disease_class.split("___")[0].replace("_", " ").title()


# ── Singleton instance ─────────────────────────────────────
classifier = DiseaseClassifier()
