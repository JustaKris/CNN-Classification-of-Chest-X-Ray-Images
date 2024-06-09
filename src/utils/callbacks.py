import datetime
import tensorflow as tf


# General purpose datetime tag
def current_time():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H-%M")

# Tensorboard callback
def tensorboard_cb(model_name):
    log_dir = f"./logs/tensorboard/{model_name}/{model_name}-{current_time()}"
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0, histogram_freq=1)

# Checkpoint callback
def checkpoint_cb(model_name):
    dir = f"./checkpoints/{model_name}/{model_name}" + "-" + "{val_weighted_accuracy:.2f}-e{epoch:02d}" + "-" + f"{current_time()}.keras"
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=dir,
        monitor='val_weighted_accuracy', 
        mode='auto', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False,
        save_freq='epoch'
        # save_freq=500
        )

# Early stopping callback
early_stoppings_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_weighted_accuracy',
    mode='max',
    verbose=True,
    patience=7,
    min_delta=0.0001, 
    restore_best_weights=True
    )

# Learning rate callback
lr_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_weighted_accuracy',
    mode='max',
    verbose=True,
    patience=3,
    mi_delta=0.0001
    )


# Testing
if __name__ == "__main__":
    print(current_time())