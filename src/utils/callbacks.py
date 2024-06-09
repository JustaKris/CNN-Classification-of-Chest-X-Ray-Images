import datetime
import tensorflow as tf


# General purpose datetime tag
def timing():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Tensorboard callback with customizable directory
def tensorboard_cb(tb_dir, model_name):
    log_dir = f"./logs/tensorboard/{model_name}/{model_name}-{timing()}"
    return tf.keras.callbacks.TensorBoard(log_dir=tb_dir, profile_batch=0)

# Checkpoint callback with customizable directory
def checkpoint_cb(model_name):
    dir = f"./checkpoints/{model_name}/{model_name}" + "-" + "{val_accuracy:.2f}-{epoch:02d}" + "-" + f"{timing()}.keras"
    return tf.keras.callbacks.ModelCheckpoint(
        dir,
        monitor='val_accuracy', 
        mode='max', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False
        )

# Early stopping callback
early_stoppings_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=True,
    patience=7,
    min_delta=0.0001, 
    restore_best_weights=True
    )

# Learning rate callback
lr_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    verbose=True,
    patience=3,
    mi_delta=0.0001
    )