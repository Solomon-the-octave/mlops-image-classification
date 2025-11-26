import tensorflow as tf
import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = "models/base_cifar10_model.h5"

def load_model():
    """Load the trained CIFAR-10 model."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_single_image(model, image_array):
    """
    Run a prediction on a single preprocessed image.
    Returns:
    - predicted_class_name
    - confidence (probability)
    """
    preds = model.predict(image_array)
    idx = np.argmax(preds)
    confidence = preds[0][idx]

    return CLASS_NAMES[idx], float(confidence)
