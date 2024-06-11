# Datasets
DATASETS = {
    "xray_train": "./data/Chest X-Rays/train",
    "xray_test": "./data/Chest X-Rays/test",
    "xray_val": "./data/Chest X-Rays/val"
}

# Classes
CLASSES = {
    0: 'COVID19',
    1: 'NORMAL',
    2: 'PNEUMONIA',
    3: 'TURBERCULOSIS'
}

# Basic model config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
COLOR_MODE = "rgb"
INPUT_SHAPE = (*IMAGE_SIZE, 1 if COLOR_MODE == "grayscale" else 3)

# Class weights manual calculation
CLASS_WEIGHTS = {
    0: 8.4, 
    1: 2.9, 
    2: 1, 
    3: 6
}
