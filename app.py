# streamlit_app/app.py

import streamlit as st
from src.models.model import load_model, predict
from src.utils import preprocess_image

# Set up the Streamlit interface
st.title("Chest X-Ray Image Classification")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image(uploaded_file)
    
    # Load the model
    model = load_model('path/to/saved_model')
    
    # Make prediction
    prediction = predict(model, image)
    
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Display the result
    st.write(f"Prediction: {prediction}")
