import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model as keras_load_model

BASE_MODEL_PATH = "models/base_cifar10_model.h5"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model(model_path: str = BASE_MODEL_PATH):
    """Load a Keras model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model file not found at: {model_path}")
    print(f"Loading model from: {model_path}")
    return keras_load_model(model_path)


def predict_single_image(model, img_array):
    """
    Predict class for a single preprocessed image.
    img_array shape must be (1, 32, 32, 3)
    """
    probs = model.predict(img_array)[0]
    idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[idx]
    confidence = probs[idx]
    return predicted_class, confidence, probs


# ----------------------------
#       RETRAINING LOGIC
# ----------------------------

def load_retraining_dataset(train_dir):
    """
    Loads user-uploaded retraining images from data/train/<class>/...
    Returns TF dataset.
    """
    img_size = (32, 32)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=32,
        shuffle=True
    )

    # Normalize
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds


def retrain_model(
    model_path=BASE_MODEL_PATH,
    train_data_dir="data/train",
    output_model_path="models/base_cifar10_model_retrained.h5",
    epochs=3,
    batch_size=32,
):
    """
    Retrains the base model using images inside data/train/
    """
    if not os.path.exists(train_data_dir):
        raise ValueError(" No training data found. Upload images first!")

    print("Loading base model...")
    model = load_model(model_path)

    print("Loading retraining dataset...")
    train_ds = load_retraining_dataset(train_data_dir)

    print("Starting retraining...")
    model.fit(train_ds, epochs=epochs)

    print(f"Saving new retrained model to: {output_model_path}")
    model.save(output_model_path)

    return output_model_path
