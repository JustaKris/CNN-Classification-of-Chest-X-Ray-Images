import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization # type: ignore
from src.config import IMAGE_SIZE, NUM_CLASSES

class EfficientNetTransfer:
    def __init__(self, input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
        
        base_model.trainable = False  # Freeze the base model initially

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def compile(self, learning_rate=1e-4):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
    
    def __call__(self):
        return self.model
