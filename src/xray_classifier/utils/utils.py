"""General-purpose utility functions for image loading, preprocessing, and model management."""

import os
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

import joblib
import numpy as np
import requests  # type: ignore[import-untyped]
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input  # type: ignore

from xray_classifier.config import COLOR_MODE, IMAGE_SIZE
from xray_classifier.exception import CustomException
from xray_classifier.logger import get_logger

logger = get_logger(__name__)


def get_date():
    """Return the current date as a YYYY.MM.DD string."""
    return datetime.now().strftime("%Y.%m.%d")


def get_time():
    """Return the current time as an HH-MM string."""
    return datetime.now().strftime("%H-%M")


def load_image_from_url(url, save_dir="./artifacts", save_filename="downloaded_image.jpg"):
    """Downloads an image from a URL and saves it locally.

    Args:
        url (str): The URL of the image to download.
        save_dir (str): The directory to save the downloaded image.
        save_filename (str): The filename to save the downloaded image.

    Returns:
        str: The local path of the downloaded image.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.error("Invalid URL scheme: %s", parsed.scheme)
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=15)

        if response.status_code == 200:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Full path to save the image
            save_path = os.path.join(save_dir, save_filename)

            # Read the image data into memory
            image_data = BytesIO(response.content)
            # Save the image locally
            with open(save_path, "wb") as f:
                f.write(image_data.getbuffer())

            logger.info("Image downloaded and saved to %s", save_path)
            return save_path
        else:
            raise requests.exceptions.RequestException(
                f"Failed to download image. Status code: {response.status_code}"
            )

    except requests.exceptions.RequestException as e:
        logger.error("Error downloading image: %s", e)
        return None


def preprocess_image(image_file, target_size=IMAGE_SIZE):
    """Preprocesses an image file for input to a model.

    Args:
        image_file (str): The file path or file-like object of the image.
        target_size (tuple): The target size for the image. Defaults to IMAGE_SIZE.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    try:
        pil_image: Image.Image = Image.open(image_file)
        if pil_image.mode.lower() != COLOR_MODE:
            pil_image = pil_image.convert(COLOR_MODE.upper())
        pil_image = pil_image.resize(target_size)  # Resize to match the model input
        image_array = np.array(pil_image)  # Convert to numpy array

        # Ensure that the image has 3 channels (RGB)
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 1:
            image_array = np.concatenate((image_array,) * 3, axis=-1)

        image_array = preprocess_input(
            image_array
        )  # Preprocess the image according to ImageNet standards
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        logger.info("Image preprocessed successfully")
        return image_array
    except Exception as e:
        logger.error("Error preprocessing image: %s", e)
        raise CustomException(e)


def get_best_model_path(directory="./saved_models"):
    """Get the path of the best model file within the specified directory based on the filename.

    Args:
        directory (str): The directory to search for model files. Defaults to "./saved_models".

    Returns:
        Tuple[str, float]: A tuple containing the path of the best model file and its corresponding score.
    """
    try:
        best_score = float("-inf")
        best_model_path = None

        # Iterate through all files and subdirectories in the specified directory
        for root, _dirs, files in os.walk(directory):
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
                            best_model_path = os.path.join(root, file)

        logger.info("Best model found: %s with score %s", best_model_path, best_score)
        return best_model_path, best_score
    except Exception as e:
        logger.error("Error finding best model: %s", e)
        raise CustomException(e)


def load_model(model_path):
    """Loads a TensorFlow Keras model from the specified path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        tf.keras.Model: The loaded model.
    """
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        logger.info("Model successfully loaded.")
        return loaded_model
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise CustomException(e)


def save_pkl(obj, filename="default_name.pkl"):
    """Save an object to a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (str): The filename to save the object.
    """
    artifacts_dir = "./artifacts"
    try:
        # Ensure the artifacts directory exists
        os.makedirs(artifacts_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # File path
        file_path = os.path.join(artifacts_dir, filename)

        # Save the object
        with open(file_path, "wb") as f:
            joblib.dump(obj, f)
        logger.info("Object saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving object to %s: %s", filename, e)
        raise CustomException(e)


def load_pkl(filename="default_name.pkl"):
    """Load an object from a pickle file.

    Args:
        filename (str): The filename to load the object from.

    Returns:
        object: The loaded object.
    """
    artifacts_dir = "./artifacts"
    try:
        file_path = os.path.join(artifacts_dir, filename)

        # Load the object
        with open(file_path, "rb") as f:
            loaded_object = joblib.load(f)
        logger.info("Object loaded from %s", file_path)
        return loaded_object
    except FileNotFoundError:
        logger.error("File %s not found in %s.", filename, artifacts_dir)
        return None
    except Exception as e:
        logger.error("Error loading object from %s: %s", filename, e)
        raise CustomException(e)


if __name__ == "__main__":
    # print("Datetime: " + get_date() + "_" + get_time())

    # model = load_model(get_best_model_path("./checkpoints/MobileNetV3Transfer")[0])
    # # image = preprocess_image("./data/Chest X-Rays/train/NORMAL/IM-0127-0001.jpeg")
    # image = preprocess_image("C:/Users/ksbon/Downloads/Cat03.jpg")
    # print("Image Processed")

    # predictions = model.predict(image)
    # print([round(pred * 100, 2) for pred in predictions.tolist()[0]])

    #
    url = "https://f.hubspotusercontent30.net/hubfs/1553939/Imported_Blog_Media/IM-0005-0001_jpeg-e1585651946797.jpg"
    img = load_image_from_url(url)
    print(img)
    preprocessed_image = preprocess_image(img)
    print(preprocessed_image)
    other_image = preprocess_image("./data/Chest X-Rays/train/NORMAL/IM-0127-0001.jpeg")
    print(other_image)
