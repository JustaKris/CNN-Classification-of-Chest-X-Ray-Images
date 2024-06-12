import kerastuner as kt
from src.models.models import MobileNetV3Transfer
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.metrics import AUC, SparseCategoricalAccuracy # type: ignore
from tensorflow.keras.applications import MobileNetV3Small # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from src.data.data_loader import load_dataset
from src.config import DATASETS, CLASS_WEIGHTS, INPUT_SHAPE, CLASSES


model_name = "MobileNetV3Transfer"

def build_model(hp):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    base_model.trainable = False  # Freeze the base model initially

    x = GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(x)

    # Define hyperparameters for tuning
    # hp_units = hp.Int('units', min_value=256, max_value=512, step=256)
    hp_units = hp.Int('units', min_value=512, max_value=1024, step=512)
    # hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.2)
    hp_dropout = 0.2
    # optimizers = hp.Choice('optimizer', values=['adam', 'sgd'])
    beta_1 = hp.Float('beta_1', min_value=0.8, max_value=1.0, step=0.05)
    

    # x = Dense(hp.Int('dense1', min_value=256, max_value=512, step=256), activation='relu')(x)
    x = Dense(hp_units, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(hp.Float('dropout1', min_value=0.3, max_value=0.5, step=0.2))(x)
    x = Dropout(hp_dropout)(x)

    # x = Dense(hp.Int('dense2', min_value=128, max_value=256, step=128), activation='relu')(x)
    x = Dense(hp_units // 2, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.3, step=0.1))(x)
    x = Dropout(hp_dropout // 2)(x)

    # x = Dense(hp.Int('dense3', min_value=64, max_value=128, step=64), activation='relu')(x)
    x = Dense(hp_units // 4, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.2, step=0.1))(x)
    x = Dropout(hp_dropout // 4)(x)
    
    # x = Dense(hp.Int('dense4', min_value=32, max_value=64, step=32), activation='relu')(x)
    x = Dense(hp_units // 8, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(hp.Float('dropout4', min_value=0.05, max_value=0.1, step=0.05))(x)
    x = Dropout(hp_dropout // 8)(x)

    outputs = Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV3Transfer")

    model.compile(
        optimizer=Adam(beta_1=beta_1),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()],
        weighted_metrics=[SparseCategoricalAccuracy()],
        )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_weighted_sparse_categorical_accuracy',
    max_trials=3,
    executions_per_trial=2,
    directory=f'logs\kerastuner\{model_name}',
    project_name="02_MoreDenseUnits_beta1"
    )

# Data loader
train = load_dataset(DATASETS["xray_train"], augment=True, shuffle=True)
test = load_dataset(DATASETS["xray_test"])
val = load_dataset(DATASETS["xray_val"])

# Search
tuner.search(
    train, 
    epochs=5,
    validation_data=test, 
    # class_weights=CLASS_WEIGHTS
    )

# Best params
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of units in the first dense layer is {best_hps.get('units')}.
The optimal dropout rate is {best_hps.get('dropout')}.
The optimal optimizer is {best_hps.get('optimizer')}.
""")

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)