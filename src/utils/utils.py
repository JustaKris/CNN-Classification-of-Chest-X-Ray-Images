# src/utils.py

import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize((224, 224))  # Resize to match the model input
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(model, image):
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class
