import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input # type: ignore
from src.config import BATCH_SIZE, COLOR_MODE, CLASS_NAMES, DATASETS, IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE

# Function to load and preprocess the dataset
def load_dataset(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_names=CLASS_NAMES, augment=False, shuffle=False):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        label_mode='int',
        color_mode=COLOR_MODE
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    # dataset = dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    # Apply preprocessing using preprocess_input
    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    
    return dataset

# Augment data function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)  # Horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    return image, label


# Testing
if __name__ == "__main__":
    # Load dataset without augmentation to access class names
    dataset_dir = DATASETS["xray_train"]
    temp_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        label_mode='int',
        color_mode=COLOR_MODE
    )

    # Print class names
    class_names = temp_dataset.class_names
    print("Class Names:", class_names)

    # Load dataset with augmentation
    train_dataset = load_dataset(dataset_dir, augment=True)

    # Print a few labels
    for images, labels in train_dataset.take(1):
        print("Labels:", labels.numpy())

    # Print the unique labels and their counts
    import numpy as np
    train_labels = np.concatenate([y for _, y in train_dataset], axis=0)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print("Label Counts:", label_counts)