import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


BASE_MODEL_PATH = "models/base_cifar10_model.h5"

# BASE_MODEL_PATH = "models/your_model_name.h5"

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGE_SIZE = (32, 32)


def load_model(model_path: str = BASE_MODEL_PATH) -> keras.Model:
    """
    Load the trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = keras.models.load_model(model_path)
    return model


def predict_single_image(
    model: keras.Model,
    preprocessed_image: np.ndarray
) -> Tuple[str, float, np.ndarray]:
    """
    Make a prediction on a single preprocessed image.

    Args:
        model: Loaded Keras model.
        preprocessed_image: Numpy array of shape (1, 32, 32, 3)

    Returns:
        predicted_class_name: str
        confidence: float
        probabilities: np.ndarray of shape (num_classes,)
    """
    probs = model.predict(preprocessed_image)[0]  # (num_classes,)
    predicted_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    return predicted_class, confidence, probs


def load_retraining_dataset(
    data_dir: str = "data/train",
    batch_size: int = 32
) -> tf.data.Dataset:
    """
    Load images from a directory structure for retraining.

    Expected folder structure:
        data/train/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                ...

    Args:
        data_dir: Directory containing class subfolders.
        batch_size: Batch size for training.

    Returns:
        A tf.data.Dataset ready for model.fit()
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True
    )

    # Normalize to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    return dataset


def retrain_model(
    model_path: str = BASE_MODEL_PATH,
    train_data_dir: str = "data/train",
    output_model_path: str = "models/base_cifar10_model_retrained.h5",
    epochs: int = 3,
    batch_size: int = 32
) -> str:
    """
    Retrain the existing model on new data and save a new version.

    Args:
        model_path: Path to the existing model (.h5).
        train_data_dir: Directory containing training images in class folders.
        output_model_path: Where to save the retrained model.
        epochs: Number of additional training epochs.
        batch_size: Batch size.

    Returns:
        Path to the new retrained model.
    """
    # 1. Load existing model (pre-trained)
    model = load_model(model_path)

    # 2. Load new training data
    train_ds = load_retraining_dataset(train_data_dir, batch_size=batch_size)

    # 3. Optional: recompile 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Retrain
    model.fit(
        train_ds,
        epochs=epochs
    )

    # 5. Ensure models directory exists
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    # 6. Save new model
    model.save(output_model_path)
    print(f"Retrained model saved to: {output_model_path}")

    return output_model_path

