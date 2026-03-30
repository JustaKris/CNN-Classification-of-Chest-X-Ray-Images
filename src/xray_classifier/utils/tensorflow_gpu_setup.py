"""CUDA environment setup and TensorFlow GPU verification."""

import os


def setup_cuda(cuda_home=None):
    """Configure CUDA environment variables for TensorFlow GPU support.

    Args:
        cuda_home: Path to CUDA installation. Falls back to CUDA_HOME env var.
    """
    if cuda_home is None:
        cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home:
        print("CUDA_HOME not set; skipping CUDA path configuration.")
        return

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.pathsep.join(
        [
            os.environ.get("PATH", ""),
            os.path.join(cuda_home, "bin"),
            os.path.join(cuda_home, "libnvvp"),
        ]
    )


def verify_gpu():
    """Print available GPUs and enable memory growth on the first one."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print("All devices:", tf.config.list_physical_devices())
    if gpus:
        print("GPUs available:", gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("No GPUs available")


if __name__ == "__main__":
    setup_cuda()
    verify_gpu()
