from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D # type: ignore
from src.config import INPUT_SHAPE, CLASSES


def MobileNetV3Transfer():
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    base_model.trainable = False  # Freeze the base model initially

    x = GlobalAveragePooling2D()(base_model.output)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV3Transfer")
    return model


def EfficientNetTransfer():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    
    base_model.trainable = False  # Freeze the base model initially

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = BatchNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.05)(x)
    # x = BatchNormalization()(x)

    output = Dense(len(CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output, name="EfficientNetTransfer")
    return model
