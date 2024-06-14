import numpy as np
import streamlit as st
from src.utils.utils import get_best_model_path, load_model, load_image_from_url, preprocess_image
from src.config import CLASSES
from src.utils.grad_cam import display_grad_heatmaps
from PIL import Image

# Set up the Streamlit interface
st.title("Chest X-Ray Image Classification")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# URL input field
image_url = st.text_input("Or enter the URL of a chest X-ray image...")

def process_image(image_file):
    # Preprocess the image
    processed_image = preprocess_image(image_file)

    # Load the model
    # model = load_model('./notebooks/Checkpoints/MobileNet_transfer/MobileNet_transfer-0.73-09-408449089731872812-b8b19d47e4f446889dd4418ea0ba3644.keras')
    # model = load_model('./checkpoints/MobileNetV3Transfer/MobileNetV3Transfer-0.92-e06-2024.06.11_14-00.keras') 
    # model = load_model('./checkpoints/EfficientNetTransfer/EfficientNetTransfer-0.90-2024.06.11.keras')
    model = load_model(get_best_model_path()[0])

    # Make prediction
    # prediction_value = predict(model, processed_image)
    prediction_value = max(model.predict(processed_image).tolist()[0])
    prediction = CLASSES[int(prediction_value)]

    # Display prediction
    st.write(f"Prediction - {prediction} - with a confidence level of {round(prediction_value * 100)}%")
    
    # Display the Grad-CAM heatmap and the superimposed image
    display_grad_heatmaps(
        model=model,
        img_path=image_file,
        last_conv_layer_name="expanded_conv_10_add"  # Replace with the correct layer name for your model
        # last_conv_layer_name="top_conv"
    )
    
    # Display the Grad-CAM image
    grad_cam_image_path = "./artifacts/grad_cam.jpg"
    grad_cam_image = Image.open(grad_cam_image_path)
    st.image(grad_cam_image, caption='Original Image with overlayed Grad-CAM heatmap', use_column_width=True)

# Process the uploaded file
if uploaded_file is not None:
    process_image(uploaded_file)

# Process the image from URL
elif image_url:
    image_file = load_image_from_url(image_url)
    if image_file is not None:
        process_image(image_file)
    else:
        st.write("Failed to load image from URL.")