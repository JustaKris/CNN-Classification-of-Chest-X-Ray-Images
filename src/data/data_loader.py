import os
import glob
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64


@dataclass
class DataGeneratorConfig:
    DATASETS = {
        "xray_train": "./data/Chest X-Rays/train",
        "xray_test": "./data/Chest X-Rays/test",
        "xray_val": "./data/Chest X-Rays/val"
    }

    train_data_path: str = os.path.join('data', 'Chest X-Rays', 'train')
    test_data_path: str = os.path.join('data', 'Chest X-Rays', 'test')
    val_data_path: str = os.path.join('data', 'Chest X-Rays', 'val')

    def list_directories(self):
        return [self.train_data_path, self.test_data_path, self.val_data_path]


class DataGenerator:
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64
    COLOR_MODE = "grayscale"


    def __init__(self, train_data_path: str = None, test_data_path: str = None, val_data_path: str = None):
        self.generator_config = DataGeneratorConfig()
        if train_data_path:
            self.generator_config.train_data_path = train_data_path
        if test_data_path:
            self.generator_config.test_data_path = test_data_path
        if val_data_path:
            self.generator_config.val_data_path = val_data_path


    def generate_data(self, train_augmentation=True):
        """
        Reads the raw train and test data, and saves the datasets with 'output' in the filenames.

        Returns:
            tuple: Paths to the processed train and test data files.
        """
        if train_augmentation:
            # Data augmentation + data generator
            train_datagen = ImageDataGenerator(
                rescale=1./255,  # Normalization
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=8,
                horizontal_flip=True
                )
        else:
            # Data generator
            train_datagen = ImageDataGenerator(rescale=1./255)

        # No augmentation applied to the test data
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Generating
        train_generator = train_datagen.flow_from_directory(
            self.generator_config.train_data_path,
            target_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            color_mode=self.COLOR_MODE,
            class_mode='sparse',
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            self.generator_config.test_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=self.COLOR_MODE,
            # class_mode='categorical',
            class_mode='sparse',
            shuffle=False
        )

        val_generator = test_datagen.flow_from_directory(
            self.generator_config.val_data_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=self.COLOR_MODE,
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
