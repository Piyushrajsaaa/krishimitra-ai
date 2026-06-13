"""
KrishiMitra AI -- Advisory API Route
POST /advisory -- Main endpoint: image + query -> disease advisory
POST /feedback -- Farmer feedback on advisory quality
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.schemas.advisory import (
    AdvisoryResponse, DiseaseInfo,
    FeedbackRequest, FeedbackResponse,
    Language
)
from app.core.config import settings
from app.core.logger import logger, audit_logger
from app.utils.image_preprocessor import preprocess_image
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# ── Load classifier with correct class count ───────────────
def get_classifier():
    """Load classifier detecting num_classes from checkpoint."""
    from app.models.disease_classifier import DiseaseClassifier, build_model, get_device
    import torch

    clf = DiseaseClassifier()
    checkpoint = Path(settings.MODEL_CHECKPOINT_PATH)

    if checkpoint.exists():
        device = get_device()
        state  = torch.load(checkpoint, map_location=device)

        # Detect num_classes from checkpoint's final layer
        final_key = [k for k in state.keys() if "weight" in k][-1]
        num_classes = state[final_key].shape[0]
        logger.info(f"Checkpoint detected {num_classes} classes")

        clf.model = build_model(num_classes=num_classes, pretrained=False)
        clf.model.load_state_dict(state)
        clf.model.to(device)
        clf.model.eval()
        clf.is_loaded = True

        # Use demo classes if not 38
        if num_classes != 38:
            from torchvision import datasets
            from pathlib import Path as P
            train_dir = P("data/processed/train")
            if train_dir.exists():
                ds = datasets.ImageFolder(str(train_dir))
                clf.classes = ds.classes
                logger.info(f"Using demo classes: {clf.classes}")
    else:
        logger.warning("No checkpoint found -- loading untrained model")
        clf.load()

    return clf


# Initialize once at module level
_classifier = None

def get_or_init_classifier():
    global _classifier
    if _classifier is None:
        _classifier = get_classifier()
    if not rag_service.is_ready:
        rag_service.initialize()
    return _classifier


@router.post("/advisory", response_model=AdvisoryResponse, tags=["Advisory"])
async def get_advisory(
    request:   Request,
    image:     UploadFile       = File(...),
    query:     Optional[str]    = Form(default=None),
    language:  str              = Form(default="en"),
    crop_name: Optional[str]    = Form(default=None),
):
    """
    Main advisory endpoint.
    Upload a crop leaf image to get disease diagnosis and treatment advice.
    """
    advisory_id = str(uuid.uuid4())
    logger.info(f"Advisory request | id={advisory_id} | lang={language}")

    if language not in ["en", "hi"]:
        language = "en"

    if image.content_type not in [
        "image/jpeg", "image/jpg", "image/png", "image/webp"
    ]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG or PNG.")

    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = preprocess_image(image_bytes)
    if tensor is None:
        raise HTTPException(status_code=400, detail="Invalid image. Upload a clear leaf photo.")

    # ── Classify ───────────────────────────────────────────
    clf        = get_or_init_classifier()
    prediction = clf.predict(tensor)

    if not prediction:
        raise HTTPException(status_code=500, detail="Classification failed. Please try again.")

    disease_class = prediction["disease_class"]
    disease_name  = prediction["disease_name"]
    confidence    = prediction["confidence"]
    escalate      = prediction["escalate_to_expert"]
    detected_crop = crop_name or clf.get_crop_from_class(disease_class)

    # ── RAG ────────────────────────────────────────────────
    context = rag_service.get_context_text(
        query=query or disease_name,
        disease_name=disease_name,
        crop_name=detected_crop,
        language=language,
        top_k=2,
    )

    # ── LLM ────────────────────────────────────────────────
    llm_result = llm_service.generate_advisory(
        disease_name=disease_name,
        confidence=confidence,
        crop_name=detected_crop,
        query=query or "What is wrong with my crop?",
        context=context,
        language=language,
        advisory_id=advisory_id,
    )

    # ── Build response ─────────────────────────────────────
    disease_info = DiseaseInfo(
        disease_name=disease_name,
        confidence=confidence,
        severity=(
            "severe"   if confidence >= 0.85 else
            "moderate" if confidence >= 0.65 else "mild"
        ),
        affected_crop=detected_crop,
    )

    disclaimer = (
        "यह सलाह AI द्वारा उत्पन्न है। गंभीर मामलों में KVK से संपर्क करें।"
        if language == "hi" else
        "This advisory is AI-generated. For severe cases, consult your local KVK."
    )

    audit_logger.info(
        f"advisory_id={advisory_id} | disease={disease_class} | "
        f"confidence={confidence:.2f} | escalate={escalate} | lang={language}"
    )

    return AdvisoryResponse(
        advisory_id=advisory_id,
        disease=disease_info,
        advisory_text=llm_result["advisory_text"],
        recommendations=llm_result["recommendations"],
        confidence_score=confidence,
        language=Language(language),
        escalate_to_expert=escalate,
        disclaimer=disclaimer,
    )


@router.post("/feedback", response_model=FeedbackResponse, tags=["Advisory"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit farmer feedback — thumbs up (1) or thumbs down (0)."""
    logger.info(f"Feedback | advisory_id={feedback.advisory_id} | rating={feedback.rating}")
    audit_logger.info(
        f"FEEDBACK | advisory_id={feedback.advisory_id} | rating={feedback.rating}"
    )
    return FeedbackResponse(
        message="Thank you for your feedback! It helps us improve KrishiMitra AI.",
        advisory_id=feedback.advisory_id,
    )
