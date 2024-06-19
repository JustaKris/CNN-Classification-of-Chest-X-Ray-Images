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


def build_model(hp):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    base_model.trainable = False  # Freeze the base model initially

    x = GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(x)

    # Define hyperparameters for tuning
    # hp_units = hp.Int('units', min_value=256, max_value=512, step=256)
    hp_units = hp.Int('units', min_value=512, max_value=1024, step=512)
    # dropout = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.2)
    # optimizers = hp.Choice('optimizer', values=['adam', 'sgd'])
    beta_1 = hp.Float('beta_1', min_value=0.8, max_value=1.0, step=0.05)
    

    x = Dense(hp_units, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(hp_units // 2, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(hp_units // 4, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(hp_units // 8, activation='relu')(x)
    x = Dropout(0.05)(x)

    outputs = Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV3Transfer")

    model.compile(
        optimizer=Adam(beta_1=beta_1),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()],
        weighted_metrics=[SparseCategoricalAccuracy()],
        )
    
    return model


def random_search_tuning(train, val, project_name, model_name, objective, max_trials, executions_per_trial, epochs):
    # Initialize Random Search
    tuner = kt.RandomSearch(
        build_model,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=f'logs\kerastuner\{model_name}',
        project_name=project_name
        )
    
    # Search
    tuner.search(
        train, 
        epochs=epochs,
        validation_data=val, 
        class_weights=CLASS_WEIGHTS
        )
    
    # Best params
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print best parameters
    print("\n|--- Best Params ---|")
    for param, value in best_hps.values.items():
        print(f"|--> {param}: {round(value, 4) if isinstance(value,( int, float)) else value}")
    print("|-------------------|\n")

    # Build the model with the optimal hyperparameters
    return tuner.hypermodel.build(best_hps)


if __name__ == "__main__":
    # Load data
    train = load_dataset(DATASETS["xray_train"], augment=True, shuffle=True)
    test = load_dataset(DATASETS["xray_test"])

    #
    model_name = "MobileNetV3Transfer"

    # Run hypeparameters search
    model = random_search_tuning(
        train=train, 
        val=test,
        objective='val_weighted_sparse_categorical_accuracy',
        max_trials=3,
        executions_per_trial=2,
        epochs=5,
        model_name=model_name,
        project_name="01_MoreDenseUnits_beta1",
    )

