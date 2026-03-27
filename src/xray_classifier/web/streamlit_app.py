"""Streamlit web interface for chest X-Ray image classification with CLIP validation."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

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

_WEB_DIR = Path(__file__).resolve().parent


def _process_image(image_file, model):
    """Preprocess, predict, and display Grad-CAM heatmap for an image."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        logger.info("Image processed successfully.")

        # Make prediction
        predictions = model.predict(processed_image, verbose=False)
        prediction_class = CLASSES[int(np.argmax(predictions))]
        prediction_value = max(predictions.tolist()[0])
        logger.info(
            "Generated prediction - %s; Predictions - %s",
            prediction_class,
            predictions.tolist()[0],
        )

        # Display prediction
        st.write(
            f"Prediction - {prediction_class} - with a confidence level of"
            f" {round(prediction_value * 100)}%"
        )

        # Display the Grad-CAM heatmap and the superimposed image
        display_grad_heatmaps(
            model=model,
            img_path=image_file,
            last_conv_layer_name="expanded_conv_10_add",
        )
        logger.info("GRAD Cam generated successfully.")

        # Display the Grad-CAM image
        grad_cam_image_path = "./artifacts/grad_cam.jpg"
        grad_cam_image = Image.open(grad_cam_image_path)
        st.image(
            grad_cam_image,
            caption="Original Image with overlayed Grad-CAM heatmap",
            use_column_width=True,
        )
        logger.info("The enhanced image has been displayed successfully.")
    except Exception as e:
        logger.error("Error processing image: %s", e)
        st.write("An error occurred while processing the image.")
        raise CustomException(str(e), sys)


def _check_and_process(image_source, model, clip_model, clip_processor):
    """Check image type with CLIP, warn if not a chest X-ray, then process."""
    # Classify the image type
    if clip_model is not None and clip_processor is not None:
        top_label = "unknown"
        confidence = 0.0
        is_chest_xray = True
        try:
            top_label, confidence, is_chest_xray = classify_image(
                image_source, clip_model, clip_processor
            )
        except Exception as e:
            logger.error("Error during image type classification: %s", e)
            is_chest_xray = True  # Proceed if classification fails

        if not is_chest_xray:
            st.warning(
                f"**This image does not appear to be a chest X-ray.** "
                f"It was detected as *{top_label}* with {round(confidence * 100)}% confidence. "
                f"Results on non-chest-X-ray images will not be meaningful."
            )
            if not st.button("Proceed Anyway"):
                return
    _process_image(image_source, model)


def main():
    """Run the Streamlit chest X-ray classification application."""
    configure_logging()
    global logger  # noqa: PLW0603
    logger = get_logger(__name__)

    # Set up the Streamlit interface with wide layout and favicon
    favicon_path = _WEB_DIR / "static" / "images" / "favicon.ico"
    st.set_page_config(
        page_title="Chest X-Ray Image Classification",
        page_icon=str(favicon_path) if favicon_path.exists() else None,
        initial_sidebar_state="expanded",
    )

    # Define CSS styles directly within the Streamlit app
    css_styles = """
    <style>
    body {
        background-size: cover;
        background-repeat: no-repeat;
    }

    .container {
        max-width: 100%;
    }
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)

    # Set up the Streamlit interface
    st.title("Chest X-Ray Image Classification")
    logger.info("Streamlit app initialized.")

    # Add a legend/info text field
    st.markdown(
        """
    This application classifies chest X-ray images using a pre-trained convolutional neural
    network (CNN). The model can identify whether an X-ray image indicates the presence of three
    ilnesses - <b>Covid19, Tunerculosis and Pneumonia</b> - with <b>Normal</b> being the output
    if neither of the three are detected.

    ### How to Use This App
    - You can upload a chest X-ray image file or provide a URL to the image.
    - The app will display the predicted class and the confidence level.
    - A Grad-CAM heatmap will be generated to highlight the regions of the image that were
      important for the prediction.

    ### Links
    - [Project Repository](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images)
    - [Research Notebook](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/blob/main/notebooks/CNN%20classification%20of%20chest%20X-Ray%20Images.ipynb)
    """
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    # URL input field
    image_url = st.text_input("Or enter the URL of a chest X-ray image...")

    # Load the model
    try:
        model_path = get_best_model_path("./saved_models")[0]
        model = load_model(model_path)
    except Exception as e:
        logger.error("Error loading model: %s", e)
        st.write("An error occurred while loading the model.")
        raise CustomException(str(e), sys)

    # Load the CLIP model for image type recognition
    clip_model_inst = None
    clip_processor_inst = None
    try:
        clip_model_inst, clip_processor_inst = load_clip_model()
    except Exception as e:
        logger.error("Error loading CLIP model: %s", e)
        st.write("An error occurred while loading the image classifier.")

    # Process the uploaded file
    if uploaded_file is not None:
        logger.info("Processing uploaded file: %s", uploaded_file.name)
        _check_and_process(uploaded_file, model, clip_model_inst, clip_processor_inst)

    # Process the image from URL
    elif image_url:
        try:
            image_file = load_image_from_url(image_url)
            if image_file is not None:
                _check_and_process(image_file, model, clip_model_inst, clip_processor_inst)
            else:
                logger.error("Failed to load image from URL.")
                st.write("Failed to load image from URL.")
                raise CustomException("Failed to load image from URL", sys)
        except Exception as e:
            logger.error("Error loading image from URL: %s", e)
            st.write("An error occurred while loading the image from URL.")
            raise CustomException(str(e), sys)


# Module-level logger placeholder (initialized in main())
logger = get_logger(__name__)

if __name__ == "__main__":
    main()
