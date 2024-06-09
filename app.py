# streamlit_app/app.py

import streamlit as st
from src.utils.utils import preprocess_image, load_model, predict
from src.config import CLASSES

# Set up the Streamlit interface
st.title("Chest X-Ray Image Classification")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image(uploaded_file)
    
    # Load the model
    model = load_model('./notebooks/Checkpoints/MobileNet_transfer/MobileNet_transfer-0.73-09-408449089731872812-b8b19d47e4f446889dd4418ea0ba3644.keras')
    
    # Make prediction
    prediction = CLASSES[int(predict(model, image))]
    
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Display the result
    st.write(f"Prediction: {prediction}")
