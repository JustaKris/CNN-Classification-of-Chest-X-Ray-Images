import os
os.environ['CUDA_HOME'] = r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5'
os.environ['PATH'] += r';C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin'
os.environ['PATH'] += r';C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/libnvvp'

import tensorflow as tf

# Verify that TensorFlow can see the GPU
gpus = tf.config.list_physical_devices('GPU')
print(tf.config.list_physical_devices())
exit()
if gpus:
    print("GPUs available: ", gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPUs available")
