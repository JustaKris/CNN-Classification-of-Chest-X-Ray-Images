from src.models.models import EfficientNetTransfer, MobileNetv3Transfer
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.metrics import AUC, SparseCategoricalAccuracy # type: ignore
from src.utils.callbacks import CustomTensorBoardCallback, tensorboard_cb, checkpoint_cb, early_stoppings_cb, lr_plateau_cb
from src.utils.utils import get_best_model_path, load_model
from src.data.data_loader import load_dataset
from src.config import BATCH_SIZE, CLASS_WEIGHTS, DATASETS

EPOCHS = 20
BETA_1 = 0.95
METRICS = [
    SparseCategoricalAccuracy(),
    # AUC(multi_label=True)
]
WEIGHTED_METRICS = METRICS.copy()

# Instantiate model
model = MobileNetv3Transfer()
model_name = "MobileNetV3Transfer"
model_path, score = get_best_model_path(".\checkpoints\MobileNetv3Transfer")

# model = EfficientNetTransfer()
# model_name = "EfficientNetTransfer"
# model_path, score = get_best_model_path(".\checkpoints\EfficientNetTransfer")

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-3, beta_1=BETA_1),
    loss=SparseCategoricalCrossentropy(),
    metrics=METRICS,
    weighted_metrics=WEIGHTED_METRICS
)
# model.summary()
# exit()

# Load model
model = load_model(model_path)
print(f"Loaded model score: {score}")

# Callbacks
callbacks = [
    # CustomTensorBoardCallback(model_name),
    tensorboard_cb(model_name), 
    checkpoint_cb(model_name), 
    early_stoppings_cb,
    lr_plateau_cb
    ]

# Data loader
train = load_dataset(DATASETS["xray_train"], augment=True, shuffle=True)
test = load_dataset(DATASETS["xray_test"])
val = load_dataset(DATASETS["xray_val"])

# Initial training with frozen base model
history = model.fit(
    train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=test,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks
)

# Fine-tuning the base model
for layer in model.layers:  
    layer.trainable = False

layers_to_train = (len(model.layers) // 3)
for layer in model.layers[-layers_to_train:]:
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5, beta_1=BETA_1),
    loss=SparseCategoricalCrossentropy(),
    metrics=METRICS,
    weighted_metrics=WEIGHTED_METRICS
)

# Continue training with unfrozen layers
history_fine_tune = model.fit(
    train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=test,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks
)