"""Flask web application for chest X-Ray image classification with Grad-CAM."""

import os
import sys
import uuid

import numpy as np
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

from xray_classifier.config import CLASSES
from xray_classifier.exception import CustomException
from xray_classifier.logger import configure_logging, get_logger
from xray_classifier.utils.grad_cam import display_grad_heatmaps
from xray_classifier.utils.image_classifier import classify_image, load_clip_model
from xray_classifier.utils.utils import (
    get_best_model_path,
    load_image_from_url,
    load_model,
    preprocess_image,
)

configure_logging()
logger = get_logger(__name__)

app = Flask(__name__)

# Load models during application startup
try:
    model_path, _ = get_best_model_path("./saved_models")
    model = load_model(model_path)
    logger.info("Model successfully loaded from %s", model_path)
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise CustomException(str(e), sys)

try:
    clip_model, clip_processor = load_clip_model()
except Exception as e:
    logger.error("Error loading CLIP model: %s", e)
    raise CustomException(str(e), sys)

ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def process_image(image_file):
    """Preprocess, predict, and generate Grad-CAM heatmap for an image."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        logger.info("Image processed successfully.")

        # Make prediction
        predictions = model.predict(processed_image, verbose=False)
        prediction_class = CLASSES[int(np.argmax(predictions))]
        prediction_value = max(predictions.tolist()[0])
        logger.info(
            "Generated prediction - %s; Predictions - %s", prediction_class, predictions.tolist()[0]
        )

        # Generate Grad-CAM heatmap with unique filename
        cam_filename = f"grad_cam_{uuid.uuid4().hex}.jpg"
        display_grad_heatmaps(
            model=model,
            img_path=image_file,
            last_conv_layer_name="expanded_conv_10_add",
            cam_filename=cam_filename,
            img_array=processed_image,
        )
        logger.info("GRAD Cam generated successfully.")

        grad_cam_image_path = os.path.join("./artifacts", cam_filename)
        return prediction_class, prediction_value, grad_cam_image_path
    except Exception as e:
        logger.error("Error processing image: %s", e)
        raise CustomException(str(e), sys)


@app.route("/artifacts/<filename>")
def send_file(filename):
    """Serve generated artifact files (e.g. Grad-CAM images)."""
    return send_from_directory(ARTIFACTS_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Handle image upload/URL, classify with CLIP, predict, and redirect."""
    error = None
    force = request.form.get("force", "false") == "true"
    request_id = uuid.uuid4().hex
    temp_file_path = os.path.join(ARTIFACTS_DIR, f"upload_{request_id}.jpg")

    # When force=true, the image was already saved from the previous request
    if force:
        temp_filename = request.form.get("temp_file", "")
        if not temp_filename or os.sep in temp_filename or "/" in temp_filename:
            return render_template("index.html", error="Invalid request.")
        temp_file_path = os.path.join(ARTIFACTS_DIR, temp_filename)
        if not os.path.isfile(temp_file_path):
            return render_template("index.html", error="Saved image not found. Please re-upload.")
    else:
        if "file" in request.files and request.files["file"]:
            file = request.files["file"]
            try:
                file.save(temp_file_path)
            except Exception as e:
                error = "An error occurred while saving the uploaded image."
                logger.error("Error saving uploaded image: %s", e)
                return render_template("index.html", error=error)
        elif "image_url" in request.form and request.form["image_url"]:
            image_url = request.form["image_url"]
            try:
                result = load_image_from_url(image_url, save_filename=f"upload_{request_id}.jpg")
                if not result:
                    error = "Failed to load image from URL."
                    return render_template("index.html", error=error)
            except Exception as e:
                error = "An error occurred while loading the image from URL."
                logger.error("Error loading image from URL: %s", e)
                return render_template("index.html", error=error)
        else:
            return render_template("index.html", error="No image provided.")

        # Check if the image is a chest X-ray
        try:
            top_label, confidence, is_chest_xray = classify_image(
                temp_file_path, clip_model, clip_processor
            )
            if not is_chest_xray:
                return render_template(
                    "warning.html",
                    detected_type=top_label,
                    confidence=round(confidence * 100),
                    temp_file=os.path.basename(temp_file_path),
                )
        except Exception as e:
            logger.error("Error during image type classification: %s", e)
            # If classification fails, proceed with prediction anyway

    # Process the image
    try:
        prediction_class, prediction_value, grad_cam_image_path = process_image(temp_file_path)
        return redirect(
            url_for(
                "prediction",
                prediction=prediction_class,
                confidence=round(prediction_value * 100),
                grad_cam_image=os.path.basename(grad_cam_image_path),
            )
        )
    except Exception as e:
        error = "An error occurred while processing the image."
        logger.error("Error processing image: %s", e)
        return render_template("index.html", error=error)


@app.route("/prediction")
def prediction():
    """Render the prediction results page."""
    prediction_class = request.args.get("prediction")
    confidence = request.args.get("confidence")
    grad_cam_image = request.args.get("grad_cam_image")
    return render_template(
        "prediction.html",
        prediction=prediction_class,
        confidence=confidence,
        grad_cam_image=grad_cam_image,
    )


def main():
    """Run the Flask development server."""
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5050)),
        debug=os.environ.get("DEBUG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
