import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from src.config import BATCH_SIZE, COLOR_MODE, CLASS_NAMES, DATASETS, IMAGE_SIZE, NUM_CLASSES

AUTOTUNE = tf.data.AUTOTUNE

# Function to preprocess and augment images
def preprocess_and_augment_image(image, label):
    # Convert the image to float32
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Apply data augmentations
    image = tf.image.random_flip_left_right(image)  # Horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    return image, label

# Function to load and preprocess the dataset
def load_dataset(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_names=CLASS_NAMES, augment=False):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        label_mode='int',
        color_mode=COLOR_MODE
    )
    
    if augment:
        dataset = dataset.map(preprocess_and_augment_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    
    return dataset


# Testing
if __name__ == "__main__":
    dataset = load_dataset(DATASETS["xray_train"])
    # print(dataset)

    import numpy as np
    train_labels = np.concatenate([y for _, y in dataset], axis=0)
    train_labels = np.array(train_labels)
    print(train_labels)

# from sklearn.utils.class_weight import compute_class_weight
# # Assuming 'y_train' contains the labels of your training data
# class_labels = range(4)  # Replace 'num_classes' with the actual number of classes
# class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
# class_weights_dict = dict(zip(class_labels, class_weights))
# print("Class weights:", class_weights_dict)
