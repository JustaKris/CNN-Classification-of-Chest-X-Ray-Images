import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import IMAGE_SIZE

from skimage.io import imread, imshow
from skimage.transform import resize

def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize(IMAGE_SIZE)  # Resize to match the model input
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def process_image(image_url):
    img = imread(image_url)
    img = resize(img, IMAGE_SIZE, preserve_range=True)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img) # scale pixels between -1 and 1
    img = tf.expand_dims(img, axis = 0)
    return img

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(model, image):
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class
