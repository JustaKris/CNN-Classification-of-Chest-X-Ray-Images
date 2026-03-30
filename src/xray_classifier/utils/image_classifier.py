"""CLIP-based zero-shot image type classifier for input validation."""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from xray_classifier.logger import get_logger

logger = get_logger(__name__)

# CLIP text descriptions for zero-shot classification
CANDIDATE_LABELS = [
    "a chest X-ray",
    "a CT scan",
    "a natural photograph",
    "a painting or drawing",
    "a selfie or portrait",
    "a screenshot of a computer screen",
    "an MRI scan",
    "a document or text",
]

CHEST_XRAY_LABEL = "a chest X-ray"
CONFIDENCE_THRESHOLD = 0.5

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_clip_model(cache_dir="./saved_models"):
    """Load the CLIP model and processor, returning them as a tuple."""
    logger.info("Loading CLIP model for image type recognition...")
    model = CLIPModel.from_pretrained(
        _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION, cache_dir=cache_dir
    )
    processor = CLIPProcessor.from_pretrained(
        _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION, cache_dir=cache_dir
    )
    logger.info("CLIP model loaded successfully.")
    return model, processor


def classify_image(image_path, clip_model, clip_processor):
    """Classify an image using CLIP zero-shot classification.

    Args:
        image_path: File path (str) or file-like object of the image.
        clip_model: The loaded CLIP model.
        clip_processor: The loaded CLIP processor.

    Returns:
        tuple: (top_label, confidence, is_chest_xray)
    """
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(text=CANDIDATE_LABELS, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    top_index = int(torch.argmax(probs))
    top_label = CANDIDATE_LABELS[top_index]
    confidence = probs[0, top_index].item()

    is_chest_xray = top_label == CHEST_XRAY_LABEL and confidence >= CONFIDENCE_THRESHOLD

    logger.info(
        "Image classification: '%s' (confidence: %.4f), is_chest_xray=%s",
        top_label,
        confidence,
        is_chest_xray,
    )

    return top_label, confidence, is_chest_xray
