import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

train_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_dataset = val_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
