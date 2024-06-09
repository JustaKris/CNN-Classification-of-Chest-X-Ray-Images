IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 4
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