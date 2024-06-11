import os
import tensorflow as tf
from datetime import datetime

METRIC = 'val_weighted_sparse_categorical_accuracy'

# General purpose datetime tag
def current_time():
    return datetime.now().strftime("%Y.%m.%d_%H-%M")

# Tensorboard callback
def tensorboard_cb(model_name):
    log_dir = f"./logs/tensorboard/{model_name}/{model_name}-{current_time()}"
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        # profile_batch=0, 
        write_images=True,
        update_freq="epoch",
        histogram_freq=1
        )

# Checkpoint callback
def checkpoint_cb(model_name, initial_value_threshold=0.9):
    # Checpoints dir and model file name setup
    dir = f"./checkpoints/{model_name}/{model_name}" + "-" + "{val_weighted_sparse_categorical_accuracy:.2f}-e{epoch:02d}" + "-" + f"{current_time()}.keras"
    # dir = f"./checkpoints/{model_name}/{model_name}" + "-{val_weighted_sparse_categorical_accuracy:.2f}.keras"

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=dir,
        monitor=METRIC, 
        mode='auto', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False,
        save_freq='epoch',
        initial_value_threshold=initial_value_threshold
        )

# Early stopping callback
early_stoppings_cb = tf.keras.callbacks.EarlyStopping(
    monitor=METRIC,
    mode='max',
    verbose=True,
    patience=7,
    min_delta=0.0001, 
    restore_best_weights=True
    )

# Learning rate callback
lr_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=METRIC,
    mode='max',
    verbose=True,
    patience=3,
    mi_delta=0.0001
    )

# Define learning rate schedule function
def lr_schedule(epoch, initial_lr=0.001):
    """
    Returns a custom learning rate that decreases at certain epochs.
    """
    if epoch < 5:
        return initial_lr
    elif epoch < 15:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

# Create a LearningRateScheduler callback
lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Custom TB callback
class CustomTensorBoardCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, log_dir_base="./logs/tensorboard"):
        super().__init__()
        self.model_name = model_name
        self.log_dir_base = log_dir_base
        self.log_dir = self.create_log_dir()
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, profile_batch=0, histogram_freq=1)
        self.best_accuracy = 0

    def create_log_dir(self, accuracy=None):
        if accuracy is None:
            return f"{self.log_dir_base}/{self.model_name}/{self.model_name}"
        else:
            return f"{self.log_dir_base}/{self.model_name}/{self.model_name}-acc{accuracy:.2f}"

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            new_log_dir = self.create_log_dir(accuracy)
            os.rename(self.log_dir, new_log_dir)
            self.log_dir = new_log_dir
            self.tensorboard_callback.log_dir = self.log_dir

    def on_train_begin(self, logs=None):
        self.tensorboard_callback.set_model(self.model)

    def on_train_end(self, logs=None):
        self.tensorboard_callback.writer.close()

    def on_batch_end(self, batch, logs=None):
        self.tensorboard_callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.tensorboard_callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.tensorboard_callback.on_epoch_end(epoch, logs)


# Testing
if __name__ == "__main__":
    print(current_time())