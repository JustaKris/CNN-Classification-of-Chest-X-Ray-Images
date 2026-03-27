"""tf.data-based data pipeline with optional augmentation and ImageNet preprocessing."""

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input  # type: ignore

from xray_classifier.config import BATCH_SIZE, CLASSES, COLOR_MODE, DATASETS, IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(
    data_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASSES.values(),
    augment=False,
    shuffle=False,
):
    """Load an image dataset from a directory with optional augmentation.

    Args:
        data_dir: Path to the root directory containing class subdirectories.
        image_size: Target (height, width) for resizing images.
        batch_size: Number of images per batch.
        class_names: Ordered list of class names.
        augment: Whether to apply random augmentation transforms.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A ``tf.data.Dataset`` of (images, labels) batches.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        label_mode="int",
        color_mode=COLOR_MODE,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return dataset


def augment_image(image, label):
    """Apply random augmentation transforms to an image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label


if __name__ == "__main__":
    import numpy as np

    dataset_dir = DATASETS["xray_train"]
    temp_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=list(CLASSES.values()),
        label_mode="int",
        color_mode=COLOR_MODE,
    )

    print("Class Names:", temp_dataset.class_names)

    train_dataset = load_dataset(dataset_dir, augment=True)

    for _images, labels in train_dataset.take(1):
        print("Labels:", labels.numpy())

    train_labels = np.concatenate([y for _, y in train_dataset], axis=0)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts, strict=False))
    print("Label Counts:", label_counts)
