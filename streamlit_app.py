"""Streamlit web interface for chest X-Ray image classification with CLIP validation."""

import sys

import numpy as np
import streamlit as st
from PIL import Image

from src.config import CLASSES
from src.exception import CustomException
from src.logger import configure_logging, get_logger
from src.utils.grad_cam import display_grad_heatmaps
from src.utils.image_classifier import classify_image, load_clip_model
from src.utils.utils import get_best_model_path, load_image_from_url, load_model, preprocess_image

configure_logging()
logger = get_logger(__name__)

# Set up the Streamlit interface with wide layout and favicon
st.set_page_config(
    page_title="Chest X-Ray Image Classification",
    page_icon="./static/images/favicon.ico",
    initial_sidebar_state="expanded",
    # layout="wide"
)

# Define CSS styles directly within the Streamlit app
css_styles = """
<style>
body {
    background-image: url("../images/background.jpg"); /* Update with the path to your background image */
    background-size: cover;
    background-repeat: no-repeat;
}

.container {
    max-width: 100%;
}
</style>
"""

# Apply the CSS styles using st.markdown with unsafe_allow_html=True
st.markdown(css_styles, unsafe_allow_html=True)

# # Load CSS
# def load_css(file_name):
#     with open(file_name) as f:
#         css_content = f'<style>\n{f.read()}\n</style>'
#     st.markdown({css_content}, unsafe_allow_html=True)
#     print(css_content)

# # Load CSS and HTML files
# load_css('./static/css/styles.css')


def process_image(image_file, model):
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
            f"Prediction - {prediction_class} - with a confidence level of {round(prediction_value * 100)}%"
        )

        # Display the Grad-CAM heatmap and the superimposed image
        display_grad_heatmaps(
            model=model,
            img_path=image_file,
            last_conv_layer_name="expanded_conv_10_add",  # Replace with the correct layer name for your model
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


# Set up the Streamlit interface
st.title("Chest X-Ray Image Classification")
logger.info("Streamlit app initialized.")

# Add a legend/info text field
st.markdown("""
This application classifies chest X-ray images using a pre-trained convolutional neural network (CNN).
The model can identify whether an X-ray image indicates the presence of three ilnesses - <b>Covid19, Tunerculosis and Pneumonia
</b> - with <b>Normal</b> being the output if neither of the three are detected.

### How to Use This App
- You can upload a chest X-ray image file or provide a URL to the image.
- The app will display the predicted class and the confidence level.
- A Grad-CAM heatmap will be generated to highlight the regions of the image that were important for the prediction.

### Links
- [Project Repository](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images)
- [Research Notebook](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/blob/main/notebooks/CNN%20classification%20of%20chest%20X-Ray%20Images.ipynb)
""")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# URL input field
image_url = st.text_input("Or enter the URL of a chest X-ray image...")

# Load the model
try:
    model_path = get_best_model_path("./models")[0]
    model = load_model(model_path)
except Exception as e:
    logger.error("Error loading model: %s", e)
    st.write("An error occurred while loading the model.")
    raise CustomException(str(e), sys)
    model = None

# Load the CLIP model for image type recognition
try:
    clip_model, clip_processor = load_clip_model()
except Exception as e:
    logger.error("Error loading CLIP model: %s", e)
    st.write("An error occurred while loading the image classifier.")
    clip_model, clip_processor = None, None


def check_and_process(image_source, model):
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
    process_image(image_source, model)


# Process the uploaded file
if uploaded_file is not None:
    logger.info("Processing uploaded file: %s", uploaded_file.name)
    check_and_process(uploaded_file, model)

# Process the image from URL
elif image_url:
    try:
        image_file = load_image_from_url(image_url)
        if image_file is not None:
            check_and_process(image_file, model)
        else:
            logger.error("Failed to load image from URL.")
            st.write("Failed to load image from URL.")
            raise CustomException("Failed to load image from URL", sys)
    except Exception as e:
        logger.error("Error loading image from URL: %s", e)
        st.write("An error occurred while loading the image from URL.")
        raise CustomException(str(e), sys)
