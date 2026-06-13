"""
KrishiMitra AI — Image Preprocessor
Handles image loading, validation, and transformation
for EfficientNet-B0 input requirements (224x224).
"""

import io
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Tuple, Optional
from app.core.logger import logger

IMAGE_SIZE      = 224
MAX_FILE_SIZE_MB = 10
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_inference_transforms() -> transforms.Compose:
    """Transforms for inference — no augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transforms() -> transforms.Compose:
    """
    Transforms for training with safe augmentation.
    hue=0 to avoid uint8 overflow bug in torchvision ColorJitter.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def validate_image(image_bytes: bytes) -> Tuple[bool, str]:
    """Validates uploaded image bytes. Returns (is_valid, error_message)."""
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"Image too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB."

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        return False, "Invalid or corrupted image file."

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ALLOWED_FORMATS:
            return False, f"Unsupported format: {img.format}. Use JPEG, PNG, or WEBP."
    except Exception:
        return False, "Could not read image format."

    return True, ""


def preprocess_image(
    image_bytes: bytes,
    training: bool = False
) -> Optional[torch.Tensor]:
    """
    Full pipeline: bytes → validated PIL Image → tensor.
    Returns tensor of shape [1, 3, 224, 224] or None if invalid.
    """
    is_valid, error_msg = validate_image(image_bytes)
    if not is_valid:
        logger.warning(f"Image validation failed: {error_msg}")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)

        transform = get_training_transforms() if training else get_inference_transforms()
        tensor = transform(img)
        tensor = tensor.unsqueeze(0)

        logger.info(f"Image preprocessed | shape: {tensor.shape}")
        return tensor

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts normalized tensor back to PIL Image (for debugging)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor.squeeze(0) * std + mean
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)
