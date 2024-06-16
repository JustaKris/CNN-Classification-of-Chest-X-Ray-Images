import os
import sys
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from src.utils.utils import get_best_model_path, load_model, load_image_from_url, preprocess_image
from src.utils.grad_cam import display_grad_heatmaps
from src.config import CLASSES
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

# Load the model
try:
    model_path, _ = get_best_model_path("./saved_models")
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

        # Return the path to the Grad-CAM image
        grad_cam_image_path = "./artifacts/grad_cam.jpg"
        return prediction_class, prediction_value, grad_cam_image_path
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise CustomException(e, sys)

@app.route('/artifacts/<filename>')
def send_file(filename):
    return send_from_directory('./artifacts', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    error = None
    if 'file' in request.files and request.files['file']:
        file = request.files['file']
        try:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join('./artifacts', file.filename)
            file.save(temp_file_path)
            
            # Pass the temporary file path to process_image
            prediction_class, prediction_value, grad_cam_image_path = process_image(temp_file_path)
            
            # Remove the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            return redirect(url_for('prediction', prediction=prediction_class, confidence=round(prediction_value * 100), grad_cam_image=os.path.basename(grad_cam_image_path)))
        except Exception as e:
            error = "An error occurred while processing the uploaded image."
    elif 'image_url' in request.form and request.form['image_url']:
        image_url = request.form['image_url']
        try:
            image_file = load_image_from_url(image_url)
            if image_file:
                prediction_class, prediction_value, grad_cam_image_path = process_image(image_file)
                return redirect(url_for('prediction', prediction=prediction_class, confidence=round(prediction_value * 100), grad_cam_image=os.path.basename(grad_cam_image_path)))
            else:
                error = "Failed to load image from URL."
        except Exception as e:
            error = "An error occurred while loading the image from URL."

    return render_template('index.html', error=error)

@app.route('/prediction')
def prediction():
    prediction_class = request.args.get('prediction')
    confidence = request.args.get('confidence')
    grad_cam_image = request.args.get('grad_cam_image')
    return render_template('prediction.html', prediction=prediction_class, confidence=confidence, grad_cam_image=grad_cam_image)

if __name__ == '__main__':
    app.run(debug=True)
