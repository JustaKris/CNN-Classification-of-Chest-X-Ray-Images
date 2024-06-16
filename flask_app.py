import os
import sys
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
from src.utils.utils import get_best_model_path, load_model, load_image_from_url, preprocess_image
from src.utils.grad_cam import display_grad_heatmaps
from src.config import CLASSES
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Load the model
try:
    model_path = get_best_model_path("./saved_models")[0]
    model = load_model(model_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise CustomException(e, sys)

def process_image(image_file):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        logging.info("Image processed successfully.")

        # Make prediction
        predictions = model.predict(processed_image, verbose=False)
        prediction_class = CLASSES[int(np.argmax(predictions))]
        prediction_value = max(predictions.tolist()[0])
        logging.info(f"Generated prediction - {prediction_class}; Predictions - {predictions.tolist()[0]}")

        # Generate Grad-CAM heatmap
        display_grad_heatmaps(
            model=model,
            img_path=image_file,
            last_conv_layer_name="expanded_conv_10_add"  # Replace with the correct layer name for your model
        )
        logging.info("GRAD Cam generated successfully.")

        # Save Grad-CAM image to static/uploads
        grad_cam_image_path = "./artifacts/grad_cam.jpg"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        grad_cam_dest = os.path.join(app.config['UPLOAD_FOLDER'], 'grad_cam.jpg')
        os.rename(grad_cam_image_path, grad_cam_dest)

        return prediction_class, prediction_value, 'grad_cam.jpg'
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise CustomException(e, sys)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_class, prediction_value, grad_cam_filename = None, None, None
    error = None
    
    if request.method == 'POST':
        if 'file' in request.files and request.files['file']:
            file = request.files['file']
            try:
                prediction_class, prediction_value, grad_cam_filename = process_image(file)
            except Exception as e:
                error = "An error occurred while processing the uploaded image."
        elif 'url' in request.form and request.form['url']:
            image_url = request.form['url']
            try:
                image_file = load_image_from_url(image_url)
                if image_file:
                    prediction_class, prediction_value, grad_cam_filename = process_image(image_file)
                else:
                    error = "Failed to load image from URL."
            except Exception as e:
                error = "An error occurred while loading the image from URL."

    return render_template('index.html', prediction=prediction_class, confidence=round(prediction_value * 100) if prediction_value else None, grad_cam_image=grad_cam_filename, error=error)

if __name__ == '__main__':
    app.run(debug=True)
