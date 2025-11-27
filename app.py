import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===== MODEL & CLASSES CONFIG =====
MODEL_PATH = "models/base_cifar10_model.h5"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


@st.cache_resource
def load_cifar10_model():
    """Load the trained CIFAR-10 model once and cache it."""
    model = load_model(MODEL_PATH)
    return model


model = load_cifar10_model()

# ===== STREAMLIT UI =====
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("CIFAR-10 Image Classification Demo")

st.write(
    "This app uses a Convolutional Neural Network trained on the CIFAR-10 "
    "dataset (32×32 colour images, 10 classes). "
    "Upload an image and the model will predict its class."
)

# Sidebar data insights
st.sidebar.header("Dataset Insights")
st.sidebar.write("**Number of classes:** 10")
st.sidebar.write("**Classes:**")
for name in CLASS_NAMES:
    st.sidebar.write(f"- {name}")

uploaded_file = st.file_uploader(
    "Upload an image (PNG or JPG). If it's not 32×32, I will resize it automatically.",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Resize to CIFAR-10 size
    image_resized = image.resize((32, 32))
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # shape (1, 32, 32, 3)

    if st.button("Predict"):
        probs = model.predict(img_batch)[0]
        idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        st.success(f"Prediction: **{pred_class}** (confidence: {confidence:.2f})")

        st.subheader("Class probabilities")
        for i, cname in enumerate(CLASS_NAMES):
            st.write(f"{cname}: {probs[i]:.3f}")
