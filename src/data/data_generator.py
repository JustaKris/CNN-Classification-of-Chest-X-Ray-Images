import os
import tensorflow as tf
from dataclasses import dataclass
from src.config import BATCH_SIZE, IMAGE_SIZE, COLOR_MODE


@dataclass
class DataGeneratorConfig:
    train_data_path: str = os.path.join('data', 'Chest X-Rays', 'train')
    test_data_path: str = os.path.join('data', 'Chest X-Rays', 'test')
    val_data_path: str = os.path.join('data', 'Chest X-Rays', 'val')

    def list_directories(self):
        return [self.train_data_path, self.test_data_path, self.val_data_path]


class DataGenerator:
    def __init__(self, train_data_path: str = None, test_data_path: str = None, val_data_path: str = None):
        self.generator_config = DataGeneratorConfig()
        if train_data_path:
            self.generator_config.train_data_path = train_data_path
        if test_data_path:
            self.generator_config.test_data_path = test_data_path
        if val_data_path:
            self.generator_config.val_data_path = val_data_path


    def generate_data(self, augment_data: bool = True):
        """
        Reads the raw train, test, and validation data and returns the data generators.

        Args:
            train_augmentation (bool): Whether to apply data augmentation to the training data.

        Returns:
            tuple: A tuple containing train, test, and validation data generators.
        """
         # Data augmentation + data generator
        if augment_data:
            # Data augmentation + data generator
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,  # Normalization
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=8,
                horizontal_flip=True
                )
        else:
            # # Data generator without augmentation
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # No augmentation applied to the test data
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Generating
        train_generator = train_datagen.flow_from_directory(
            self.generator_config.train_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            class_mode='sparse',
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            self.generator_config.test_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            # class_mode='categorical',
            class_mode='sparse',
            shuffle=False
        )

        val_generator = test_datagen.flow_from_directory(
            self.generator_config.val_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=COLOR_MODE,
            # class_mode='categorical',
            class_mode='sparse',
            shuffle=False
        )
        
        return train_generator, test_generator, val_generator


if __name__ == "__main__":
    # Get generators
    data_loader = DataGenerator()
    train_generator, test_generator, val_generator = data_loader.generate_data()

    # Check image size
    print("Each image is of size: {}".format(data_loader.IMAGE_SIZE))
