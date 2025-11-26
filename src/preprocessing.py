import numpy as np
from PIL import Image

IMG_SIZE = (32, 32)   # CIFAR-10 model uses 32x32 images

def preprocess_image(image_file):
    """
    Preprocess an uploaded image for model prediction.
    
    Steps:
    - Load image
    - Resize to 32x32
    - Convert to numpy array
    - Normalize to [0,1]
    - Add batch dimension
    """
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)

    return img_array

