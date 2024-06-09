import tensorflow as tf

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Paths
train_dir = './data/Chest X-Rays/train'
val_dir = './data/Chest X-Rays/val'
test_dir = './data/Chest X-Rays/test'

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Print class names to verify
class_names = train_dataset.class_names
print(class_names)
