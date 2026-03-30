"""Pre-download CLIP model weights so they are available offline at runtime."""

from transformers import CLIPModel, CLIPProcessor

from xray_classifier.utils.image_classifier import (
    _CLIP_MODEL_NAME,
    _CLIP_MODEL_REVISION,
)

CACHE_DIR = "./saved_models"


def download_clip_model():
    """Download and cache the CLIP model and processor to saved_models/."""
    print(f"Downloading CLIP model '{_CLIP_MODEL_NAME}' to {CACHE_DIR} ...")
    CLIPModel.from_pretrained(
        _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION, cache_dir=CACHE_DIR
    )
    CLIPProcessor.from_pretrained(
        _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION, cache_dir=CACHE_DIR
    )
    print("CLIP model downloaded successfully.")


if __name__ == "__main__":
    download_clip_model()
