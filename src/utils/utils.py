import os
import dill
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from datetime import datetime
from src.config import COLOR_MODE, IMAGE_SIZE
from tensorflow.keras.applications.imagenet_utils import preprocess_input # type: ignore


# General purpose date and time tags
def get_date():
    return datetime.now().strftime("%Y.%m.%d")

def get_time():
    return datetime.now().strftime("%H-%M")

def load_image_from_url(url, save_dir='./artifacts', save_filename='downloaded_image.jpg'):
    """
    Downloads an image from a URL and saves it locally.

    Args:
        url (str): The URL of the image to download.
        save_dir (str): The directory to save the downloaded image.
        save_filename (str): The filename to save the downloaded image.

    Returns:
        str: The local path of the downloaded image.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Full path to save the image
            save_path = os.path.join(save_dir, save_filename)

            # Read the image data into memory
            image_data = BytesIO(response.content)
            # Save the image locally
            with open(save_path, 'wb') as f:
                f.write(image_data.getbuffer())

            return save_path
        else:
            raise requests.exceptions.RequestException(
                f"Failed to download image. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def preprocess_image(image_file, target_size=IMAGE_SIZE):
    """
    Preprocesses an image file for input to a model.

    Args:
        image_file (str): The file path or file-like object of the image.
        target_size (tuple): The target size for the image. Defaults to IMAGE_SIZE.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    image = Image.open(image_file)
    # image = image_file
    if image.mode.lower() != COLOR_MODE:
        image = image.convert(COLOR_MODE.upper())
    image = image.resize(target_size)  # Resize to match the model input
    image = np.array(image)  # Convert to numpy array
    
    # Ensure that the image has 3 channels (RGB)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate((image,) * 3, axis=-1)
    
    image = preprocess_input(image)  # Preprocess the image according to ImageNet standards
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_best_model_path(directory=".\checkpoints"):
    """
    Get the path of the best model file within the specified directory based on the filename.

    Args:
        directory (str): The directory to search for model files. Defaults to ".\checkpoints".

    Returns:
        Tuple[str, float]: A tuple containing the path of the best model file and its corresponding score.
    """
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

    return best_model_directory, best_score

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def save_pkl(object, filename='default_name.pkl'):
    # Ensure the artifacts directory exists
    dir = "./artifacts"
    os.makedirs(dir, exist_ok=True)  # Create the directory if it doesn't exist

    # File path
    file_path = os.path.join(dir, filename)
    
    # Save the object
    with open(file_path, 'wb') as f:
        dill.dump(object, f)
    print(f"Object saved to {file_path}")

def load_pkl(filename='default_name.pkl'):
    dir = "./artifacts"
    file_path = os.path.join(dir, filename)
    
    # Load the object
    with open(file_path, 'rb') as f:
        loaded_object = dill.load(f)
    print(f"Object loaded from {file_path}")
    return loaded_object


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