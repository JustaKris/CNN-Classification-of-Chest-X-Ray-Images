import kerastuner as kt
from src.models.models import MobileNetv3Transfer
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.metrics import AUC, SparseCategoricalAccuracy # type: ignore
from tensorflow.keras.applications import MobileNetV3Small # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D # type: ignore
from src.data.data_loader import load_dataset
from src.config import DATASETS, CLASS_WEIGHTS, INPUT_SHAPE, CLASSES


model_name = "MobileNetV3Transfer"
def build_model(hp):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    base_model.trainable = False  # Freeze the base model initially

    x = GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(x)

    # Define hyperparameters for tuning
    # hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    # hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    # hp_class_weights = hp.Choice('class_weights', values=[None, CLASS_WEIGHTS])
    # optimizers = hp.Choice('optimizer', values=['adam', 'sgd'])

    x = Dense(hp.Int('dense1', values=[512, 256]), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout1', values=[0.5, 0.3]))(x)

    x = Dense(hp.Int('dense2', values=[512, 256]) // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout2', values=[0.5, 0.3]) - 0.1)(x)

    x = Dense(hp.Int('dense3', values=[512, 256]) // 4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout3', values=[0.5, 0.3]) - 0.2)(x)
    
    x = Dense(hp.Int('dense4', values=[512, 256]) // 8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout4', values=[0.5, 0.3]) - 0.3)(x)

    outputs = Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()],
        weighted_metrics=[SparseCategoricalAccuracy()],
        # class_weight=hp_class_weights
        )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_weighted_sparse_categorical_accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory=f'logs\kerastuner\{model_name}',
    project_name=model_name
    )

# Data loader
train = load_dataset(DATASETS["xray_train"], augment=True, shuffle=True)
test = load_dataset(DATASETS["xray_test"])
val = load_dataset(DATASETS["xray_val"])

# Search
tuner.search(train, epochs=10, validation_data=test, class_weights=CLASS_WEIGHTS)

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