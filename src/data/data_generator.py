"""ImageDataGenerator-based data pipeline for chest X-Ray datasets."""

import os
from dataclasses import dataclass

import tensorflow as tf

from src.config import BATCH_SIZE, COLOR_MODE, IMAGE_SIZE


@dataclass
class DataGeneratorConfig:
    """File-system paths for train / test / val splits."""

    train_data_path: str = os.path.join("data", "Chest X-Rays", "train")
    test_data_path: str = os.path.join("data", "Chest X-Rays", "test")
    val_data_path: str = os.path.join("data", "Chest X-Rays", "val")

    def list_directories(self):
        """Return all data split directories as a list."""
        return [self.train_data_path, self.test_data_path, self.val_data_path]


class DataGenerator:
    """Create Keras ImageDataGenerators with optional augmentation."""

    def __init__(
        self,
        train_data_path: str = None,
        test_data_path: str = None,
        val_data_path: str = None,
    ):
        """Initialize with optional custom data paths."""
        self.generator_config = DataGeneratorConfig()
        if train_data_path:
            self.generator_config.train_data_path = train_data_path
        if test_data_path:
            self.generator_config.test_data_path = test_data_path
        if val_data_path:
            self.generator_config.val_data_path = val_data_path

    def generate_data(self, augment_data: bool = True):
        """Read train/test/val data and return generators.

        Args:
            augment_data: Whether to apply data augmentation to the training data.

        Returns:
            A tuple of (train_generator, test_generator, val_generator).
        """
        if augment_data:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=8,
                horizontal_flip=True,
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            self.generator_config.train_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            class_mode="sparse",
            shuffle=True,
        )

        test_generator = test_datagen.flow_from_directory(
            self.generator_config.test_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            class_mode="sparse",
            shuffle=False,
        )

        val_generator = test_datagen.flow_from_directory(
            self.generator_config.val_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            class_mode="sparse",
            shuffle=False,
        )

        return train_generator, test_generator, val_generator


if __name__ == "__main__":
    data_loader = DataGenerator()
    train_generator, test_generator, val_generator = data_loader.generate_data()
    print(f"Each image is of size: {IMAGE_SIZE}")
