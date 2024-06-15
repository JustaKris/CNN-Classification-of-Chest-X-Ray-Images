import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from src.config import CLASSES
from IPython.display import display, Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="./artifacts/", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")  # Updated method to get colormap

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    os.makedirs(cam_path, exist_ok=True)
    superimposed_img_path = os.path.join(cam_path, "grad_cam.jpg")
    superimposed_img.save(superimposed_img_path)

    # Display Grad CAM in Jupyter or IPython
    # display(Image(filename=superimposed_img_path))

def display_grad_heatmaps(model, img_path, last_conv_layer_name, display_heatmap=0):
    # Prepare image
    preprocess_input = tf.keras.applications.imagenet_utils.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=(224, 224)))  # Ensure size matches model input size

    # Target final convolutional layer
    last_conv_layer_name = last_conv_layer_name

    # Remove softmax activation from final dense layer
    model.layers[-1].activation = None

    # Predict image
    preds = model.predict(img_array, verbose=False)

    # Print predicted class
    print("Predicted:", list(CLASSES.values())[np.argmax(preds)])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    if display_heatmap:
        # Display heatmap
        plt.figure(figsize=(9, 9))
        plt.matshow(heatmap, fignum=1)
        plt.axis("off")
        plt.show()
    else:
        # Show GRAD heatmap
        save_and_display_gradcam(img_path, heatmap)

    # Re-add the softmax activation for future predictions
    model.layers[-1].activation = tf.keras.activations.softmax

def get_img_array(img_path, size):
    # `img` is a PIL image of size 224x224
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    return array

if __name__ == "__main__":
    import matplotlib.image as mpimg
    from src.utils.utils import preprocess_image, load_model, get_best_model_path

    # Load the model
    model = load_model(get_best_model_path()[0])
    image_path = "./data/Chest X-Rays/train/NORMAL/IM-0127-0001.jpeg"
    image = preprocess_image(image_path)
    print("Image Processed\n")

    predictions = model.predict(image, verbose=False)
    print(f"Predictions -> {[round(pred * 100, 2) for pred in predictions.tolist()[0]]}\n")

    # Display GRAD heatmaps
    display_grad_heatmaps(
        model=model,
        img_path=image_path,
        last_conv_layer_name="expanded_conv_10_add",  # Replace with the correct layer name for your model
    )

    # Path to the image file
    image_path = "./artifacts/grad_cam.jpg"

    # Read the image
    image = mpimg.imread(image_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()
