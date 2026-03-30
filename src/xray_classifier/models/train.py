"""Training script for transfer learning models on chest X-ray data."""

from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from xray_classifier.config import BATCH_SIZE, CLASS_WEIGHTS, CLASSES, DATASETS
from xray_classifier.data.data_loader import load_dataset
from xray_classifier.models.models import EfficientNetTransfer
from xray_classifier.utils.callbacks import (
    checkpoint_cb,
    early_stoppings_cb,
    lr_plateau_cb,
    tensorboard_cb,
)
from xray_classifier.utils.evaluate import evaluate_model, plot_accuracy_loss
from xray_classifier.utils.utils import get_best_model_path

EPOCHS = 20
BETA_1 = 0.9
METRICS = [
    SparseCategoricalAccuracy(),
]


def main():
    """Train an EfficientNet transfer learning model on chest X-ray data."""
    model = EfficientNetTransfer()
    model_path, score = get_best_model_path("./checkpoints/EfficientNetTransfer")

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3, beta_1=BETA_1),
        loss=SparseCategoricalCrossentropy(),
        metrics=METRICS,
        weighted_metrics=METRICS,
    )

    # Retrieve model name
    model_name = model.name
    print(model_name)

    # Callbacks
    callbacks = [
        tensorboard_cb(model_name),
        checkpoint_cb(model_name),
        early_stoppings_cb,
        lr_plateau_cb,
    ]

    # Data loader
    train = load_dataset(DATASETS["xray_train"], augment=True, shuffle=True)
    test = load_dataset(DATASETS["xray_test"])
    # val = load_dataset(DATASETS["xray_val"])

    # Initial training with frozen base model
    history = model.fit(
        train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=test,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
    )

    # Eval
    evaluate_model(model, test, CLASSES.values())

    # Fine-tuning the base model
    layers_to_train = len(model.layers) // 2
    for layer in model.layers[-layers_to_train:]:
        layer.trainable = True
    print(f"Model layers for fine tuning - Bottom {layers_to_train}")

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-4, beta_1=BETA_1),
        loss=SparseCategoricalCrossentropy(),
        metrics=METRICS,
        weighted_metrics=METRICS,
    )

    # Continue training with unfrozen layers
    # history_fine_tune = model.fit(
    #     train,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     validation_data=test,
    #     class_weight=CLASS_WEIGHTS,
    #     callbacks=callbacks,
    # )

    # Eval
    evaluate_model(model, test, CLASSES.values())
    plot_accuracy_loss(history=history.history, metric="weighted_sparse_categorical_accuracy")


if __name__ == "__main__":
    main()
