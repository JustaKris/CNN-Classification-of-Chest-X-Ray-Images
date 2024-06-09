import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import IMAGE_SIZE

from skimage.io import imread, imshow
from skimage.transform import resize

def preprocess_image(image_file):
    image = Image.open(image_file)
    if image.mode.upper() != 'RGB':
        image = image.convert('RGB')
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

def get_best_model_path(directory=".\checkpoints"):
    best_score = float('-inf')
    best_model_directory = None

    # Iterate through all files and subdirectories in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a model file
            if file.endswith(".keras"):
                # Extract the score from the file name
                filename_parts = file.split("-")
                if len(filename_parts) >= 2:
                    score = float(filename_parts[1])
                    # Update the best score and directory if a better model is found
                    if score > best_score:
                        best_score = score
                        best_model_directory = os.path.join(root, file)

    return best_model_directory


if __name__ == "__main__":
    print(get_best_model_path())