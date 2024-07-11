import sys
import numpy as np
import streamlit as st
from PIL import Image
from src.utils.utils import get_best_model_path, load_model, load_image_from_url, preprocess_image
from src.utils.grad_cam import display_grad_heatmaps
from src.config import CLASSES
from src.logger import logging
from src.exception import CustomException

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
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        logging.info("Image processed successfully.")

        # Make prediction
        predictions = model.predict(processed_image, verbose=False)
        prediction_class = CLASSES[int(np.argmax(predictions))]
        prediction_value = max(predictions.tolist()[0])
        logging.info(f"Generated prediction - {prediction_class}; Predictions - {predictions.tolist()[0]}")

        # Display prediction
        st.write(f"Prediction - {prediction_class} - with a confidence level of {round(prediction_value * 100)}%")

        # Display the Grad-CAM heatmap and the superimposed image
        display_grad_heatmaps(
            model=model,
            img_path=image_file,
            last_conv_layer_name="expanded_conv_10_add"  # Replace with the correct layer name for your model
        )
        logging.info("GRAD Cam generated successfully.")

        # Display the Grad-CAM image
        grad_cam_image_path = "./artifacts/grad_cam.jpg"
        grad_cam_image = Image.open(grad_cam_image_path)
        st.image(grad_cam_image, caption='Original Image with overlayed Grad-CAM heatmap', use_column_width=True)
        logging.info("The enhanced image has been displayed successfully.")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.write("An error occurred while processing the image.")
        raise CustomException(e, sys)


# Set up the Streamlit interface
st.title("Chest X-Ray Image Classification")
logging.info("Streamlit app initialized.")

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
    model_path = get_best_model_path("./saved_models")[0]
    model = load_model(model_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.write("An error occurred while loading the model.")
    raise CustomException(e, sys)
    model = None

# Process the uploaded file
if uploaded_file is not None:
    logging.info(f"Processing uploaded file: {uploaded_file.name}")
    process_image(uploaded_file, model)

# Process the image from URL
elif image_url:
    try:
        image_file = load_image_from_url(image_url)
        if image_file is not None:
            process_image(image_file, model)
        else:
            logging.error("Failed to load image from URL.")
            st.write("Failed to load image from URL.")
            raise CustomException("Failed to load image from URL", sys)
    except Exception as e:
        logging.error(f"Error loading image from URL: {e}")
        st.write("An error occurred while loading the image from URL.")
        raise CustomException(e, sys)