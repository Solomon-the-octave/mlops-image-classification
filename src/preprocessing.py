import numpy as np
from PIL import Image

def preprocess_image(uploaded_file):
    """
    Preprocess an uploaded image for prediction.
    Returns a numpy array of shape (1, 32, 32, 3)
    """
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
