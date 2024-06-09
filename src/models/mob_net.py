# src/models/mobilenet_model.py
import tensorflow as tf
from src.config import IMAGE_SIZE

class MobileNetTransfer:
    LEARNING_RATE = 1e-7
    BETA_1 = 0.95
    EPOCHS = 30

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        base_model = tf.keras.applications.MobileNetV3Small(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
        base_model.trainable = True

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, beta_1=self.BETA_1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def get_model(self):
        return self.model
