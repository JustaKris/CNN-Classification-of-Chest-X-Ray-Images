# Basic model config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
COLOR_MODE = "rgb"
INPUT_SHAPE = (*IMAGE_SIZE, 1 if COLOR_MODE == "grayscale" else 3)

# Datasets and classes
DATASETS = {
    "xray_train": "./data/Chest X-Rays/train",
    "xray_test": "./data/Chest X-Rays/test",
    "xray_val": "./data/Chest X-Rays/val"
}
CLASSES = {
    # 'COVID19': 0,
    # 'NORMAL': 1,
    # 'PNEUMONIA': 2,
    # 'TURBERCULOSIS': 3,
    0: 'COVID19',
    1: 'NORMAL',
    2: 'PNEUMONIA',
    3: 'TURBERCULOSIS'
}
NUM_CLASSES = len(CLASSES)
# CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
CLASS_WEIGHTS = {
    0: 8.4, 
    1: 2.9, 
    2: 1, 
    3: 6
} # Manual calculation