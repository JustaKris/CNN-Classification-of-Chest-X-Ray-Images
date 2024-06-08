import tensorflow as tf

def setup_tensorboard(log_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def log_tensorboard(log_dir, model, data):
    tensorboard_callback = setup_tensorboard(log_dir)
    
    model.fit(data, epochs=5, callbacks=[tensorboard_callback])  # Example usage
