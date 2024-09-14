# import torch

# if torch.cuda.is_available():
#     print("CUDA is available. PyTorch is using GPU.")
#     print(f"Device name: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available. PyTorch is using CPU.")

import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print("GPU:", gpu)
else:
    print("No GPU detected, running on CPU")
