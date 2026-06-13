"""
KrishiMitra AI — Unit Tests for Disease Classifier
Run: pytest tests/unit/test_model.py -v
"""

import pytest
import torch
import io
from PIL import Image
from app.models.disease_classifier import (
    build_model,
    DiseaseClassifier,
    DISEASE_CLASSES,
    DISPLAY_NAMES,
)
from app.utils.image_preprocessor import (
    preprocess_image,
    validate_image,
    get_inference_transforms,
    get_training_transforms,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def dummy_image_bytes():
    """Creates a valid dummy JPEG image in memory."""
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_tensor():
    """Creates a dummy preprocessed tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def loaded_classifier():
    """Creates a DiseaseClassifier with untrained model."""
    clf = DiseaseClassifier()
    clf.model = build_model(num_classes=38, pretrained=False)
    clf.model.eval()
    clf.is_loaded = True
    return clf


# ── Disease Classes Tests ─────────────────────────────────

def test_disease_classes_count():
    """PlantVillage has exactly 38 classes."""
    assert len(DISEASE_CLASSES) == 38


def test_disease_classes_unique():
    """All class names must be unique."""
    assert len(DISEASE_CLASSES) == len(set(DISEASE_CLASSES))


def test_display_names_generated():
    """Every class has a display name."""
    assert len(DISPLAY_NAMES) == 38
    for cls in DISEASE_CLASSES:
        assert cls in DISPLAY_NAMES


def test_healthy_classes_present():
    """Dataset must include healthy crop classes."""
    healthy = [c for c in DISEASE_CLASSES if "healthy" in c.lower()]
    assert len(healthy) > 0


# ── Model Architecture Tests ──────────────────────────────

def test_model_builds_38_classes():
    """Model output matches number of disease classes."""
    model = build_model(num_classes=38, pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 38)


def test_model_builds_custom_classes():
    """Model can be built with custom class count."""
    model = build_model(num_classes=5, pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 5)


def test_model_output_is_logits():
    """Raw model output should be logits (not probabilities)."""
    model = build_model(num_classes=38, pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    # Logits can be any value; softmax output sums to 1
    probs = torch.softmax(out, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_base_layers_frozen():
    """Base EfficientNet layers must be frozen (transfer learning)."""
    model = build_model(num_classes=38, pretrained=False)
    frozen = [p for name, p in model.named_parameters()
              if "classifier" not in name and not p.requires_grad]
    assert len(frozen) > 0


def test_classifier_head_trainable():
    """Custom classifier head must be trainable."""
    model = build_model(num_classes=38, pretrained=False)
    trainable = [p for p in model.classifier.parameters() if p.requires_grad]
    assert len(trainable) > 0


# ── Preprocessor Tests ────────────────────────────────────

def test_valid_image_passes_validation(dummy_image_bytes):
    """Valid JPEG should pass validation."""
    is_valid, msg = validate_image(dummy_image_bytes)
    assert is_valid
    assert msg == ""


def test_invalid_bytes_fail_validation():
    """Random bytes should fail validation."""
    is_valid, msg = validate_image(b"not_an_image_bytes_xyz")
    assert not is_valid


def test_oversized_image_fails_validation():
    """Image over 10MB should fail validation."""
    big_bytes = b"x" * (11 * 1024 * 1024)
    is_valid, msg = validate_image(big_bytes)
    assert not is_valid
    assert "large" in msg.lower()


def test_preprocess_returns_correct_shape(dummy_image_bytes):
    """Preprocessed tensor must be [1, 3, 224, 224]."""
    tensor = preprocess_image(dummy_image_bytes)
    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_returns_float_tensor(dummy_image_bytes):
    """Tensor must be float32."""
    tensor = preprocess_image(dummy_image_bytes)
    assert tensor.dtype == torch.float32


def test_preprocess_invalid_returns_none():
    """Invalid image bytes must return None."""
    result = preprocess_image(b"invalid")
    assert result is None


# ── Classifier Predict Tests ──────────────────────────────

def test_predict_returns_dict(loaded_classifier, dummy_tensor):
    """Prediction must return a dictionary."""
    result = loaded_classifier.predict(dummy_tensor)
    assert isinstance(result, dict)


def test_predict_has_required_keys(loaded_classifier, dummy_tensor):
    """Prediction must have all required keys."""
    result = loaded_classifier.predict(dummy_tensor)
    required_keys = [
        "disease_class", "disease_name", "confidence",
        "top5", "escalate_to_expert", "is_healthy"
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_predict_confidence_in_range(loaded_classifier, dummy_tensor):
    """Confidence score must be between 0 and 1."""
    result = loaded_classifier.predict(dummy_tensor)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_top5_has_five_items(loaded_classifier, dummy_tensor):
    """Top-5 predictions must return exactly 5 items."""
    result = loaded_classifier.predict(dummy_tensor)
    assert len(result["top5"]) == 5


def test_predict_without_loading_returns_empty():
    """Calling predict before load() must return empty dict."""
    clf = DiseaseClassifier()
    result = clf.predict(torch.randn(1, 3, 224, 224))
    assert result == {}


def test_get_crop_from_class(loaded_classifier):
    """Crop name extraction must work correctly."""
    crop = loaded_classifier.get_crop_from_class("Tomato___Early_blight")
    assert crop == "Tomato"
