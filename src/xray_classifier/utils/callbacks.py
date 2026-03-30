"""Keras training callbacks for TensorBoard, checkpoints, and learning rate scheduling."""

import os

import tensorflow as tf
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from xray_classifier.utils.utils import get_date, get_time

METRIC = "val_weighted_sparse_categorical_accuracy"


def tensorboard_cb(model_name):
    """Create a TensorBoard callback with a timestamped log directory."""
    log_dir = f"./logs/tensorboard/{model_name}/{model_name}-{get_date()}_{get_time()}"
    return TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq="epoch",
        histogram_freq=1,
    )


def checkpoint_cb(model_name, initial_value_threshold=0.9):
    """Create a ModelCheckpoint callback that saves the best model per epoch."""
    filepath = (
        f"./checkpoints/{model_name}/{model_name}"
        + "-{val_weighted_sparse_categorical_accuracy:.2f}"
        + f"-{get_date()}.keras"
    )

    return ModelCheckpoint(
        filepath=filepath,
        monitor=METRIC,
        mode="auto",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
        initial_value_threshold=initial_value_threshold,
    )


early_stoppings_cb = EarlyStopping(
    monitor=METRIC,
    mode="max",
    verbose=1,
    patience=8,
    min_delta=0.001,
    restore_best_weights=True,
)

lr_plateau_cb = ReduceLROnPlateau(
    monitor=METRIC,
    mode="max",
    verbose=1,
    patience=3,
    min_delta=0.0001,
)


def lr_schedule(epoch, initial_lr=0.001):
    """Return a custom learning rate that decreases at certain epochs."""
    if epoch < 5:
        return initial_lr
    elif epoch < 15:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01


lr_scheduler_cb = LearningRateScheduler(lr_schedule)


class CustomTensorBoardCallback(tf.keras.callbacks.Callback):
    """TensorBoard callback that renames log directories based on accuracy."""

    def __init__(self, model_name, log_dir_base="./logs/tensorboard"):
        """Initialize with model name and base log directory."""
        super().__init__()
        self.model_name = model_name
        self.log_dir_base = log_dir_base
        self.log_dir = self.create_log_dir()
        self.tensorboard_callback = TensorBoard(
            log_dir=self.log_dir,
            profile_batch=0,
            histogram_freq=1,
        )
        self.best_accuracy = 0

    def create_log_dir(self, accuracy=None):
        """Generate the log directory path, optionally including accuracy."""
        if accuracy is None:
            return f"{self.log_dir_base}/{self.model_name}/{self.model_name}"
        else:
            return f"{self.log_dir_base}/{self.model_name}/{self.model_name}-acc{accuracy:.2f}"

    def on_epoch_end(self, epoch, logs=None):
        """Rename log dir on accuracy improvement and delegate to TensorBoard."""
        if logs is None:
            logs = {}
        accuracy = logs.get("accuracy")
        if accuracy and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            new_log_dir = self.create_log_dir(accuracy)
            os.rename(self.log_dir, new_log_dir)
            self.log_dir = new_log_dir
            self.tensorboard_callback.log_dir = self.log_dir  # type: ignore[attr-defined]
        self.tensorboard_callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        """Set the model on the inner TensorBoard callback."""
        self.tensorboard_callback.set_model(self.model)

    def on_train_end(self, logs=None):
        """Close the TensorBoard writer."""
        self.tensorboard_callback.writer.close()  # type: ignore[attr-defined]

    def on_batch_end(self, batch, logs=None):
        """Delegate batch end to the inner TensorBoard callback."""
        self.tensorboard_callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Delegate epoch begin to the inner TensorBoard callback."""
        self.tensorboard_callback.on_epoch_begin(epoch, logs)
