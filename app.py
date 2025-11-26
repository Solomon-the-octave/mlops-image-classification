import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_image
from src.model import (
    load_model,
    predict_single_image,
    retrain_model,
    BASE_MODEL_PATH
)

# ----------------- Streamlit config -----------------

st.set_page_config(
    page_title="MLOps Image Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MLOps Image Classification Demo")
st.write(
    """
    This app demonstrates an end-to-end image classification pipeline with:
    - Model prediction
    - Data upload for retraining
    - Triggering a retrain of the model
    - Basic data insights
    """
)

# CIFAR-10 class names (same as in your notebook / model.py)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


@st.cache_resource
def get_model(model_path: str = BASE_MODEL_PATH):
    """Load and cache the trained model."""
    model = load_model(model_path)
    return model


model = get_model()

tab_predict, tab_retrain, tab_insights = st.tabs(
    [" Predict", "🔁 Upload & Retrain", "📊 Insights"]
)

# ----------------- TAB 1: PREDICTION -----------------

with tab_predict:
    st.subheader(" Predict a Single Image")

    uploaded_file = st.file_uploader(
        "Upload an image (any object similar to CIFAR-10 classes)",
        type=["jpg", "jpeg", "png"],
        key="predict_uploader"
    )

    if uploaded_file is not None:
        # Show the uploaded image
        st.image(uploaded_file, caption="Uploaded image", use_column_width=False, width=200)

        # Preprocess and predict
        img_array = preprocess_image(uploaded_file)
        predicted_class, confidence, probs = predict_single_image(model, img_array)

        st.success(f"Predicted class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(CLASS_NAMES)), probs)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)
    else:
        st.info("Please upload an image to see the prediction.")


# ----------------- TAB 2: UPLOAD & RETRAIN -----------------

with tab_retrain:
    st.subheader("Upload New Training Data & Trigger Retrain")

    st.write(
        """
        Use this section to upload new labeled images that will be used
        to retrain the model.
        - Choose a **class name** (e.g. `cat`, `dog`, `car`, etc.)
        - Upload multiple images for that class.
        """
    )

    class_name = st.text_input(
        "Class name for these images (will be used as folder name under `data/train/`)",
        value="custom_class"
    )

    train_files = st.file_uploader(
        "Upload training images for this class",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key="retrain_uploader"
    )

    save_status = st.empty()

    if st.button(" Save Uploaded Images"):
        if not train_files:
            st.warning("Please upload at least one image.")
        else:
            save_dir = os.path.join("data", "train", class_name)
            os.makedirs(save_dir, exist_ok=True)

            saved_paths = []
            for f in train_files:
                dest_path = os.path.join(save_dir, f.name)
                with open(dest_path, "wb") as out_file:
                    out_file.write(f.read())
                saved_paths.append(dest_path)

            save_status.success(
                f"Saved {len(saved_paths)} images to `{save_dir}`. These will be used for retraining."
            )

    st.markdown("---")

    st.write(
        """
        After uploading and saving images into the `data/train/` directory,
        you can trigger a retrain of the model.
        """
    )

    if st.button(" Trigger Model Retraining"):
        with st.spinner("Retraining model on data in `data/train/`..."):
            new_model_path = retrain_model(
                model_path=BASE_MODEL_PATH,
                train_data_dir="data/train",
                output_model_path="models/base_cifar10_model_retrained.h5",
                epochs=3,
                batch_size=32
            )
            # Reload cached model
            st.cache_resource.clear()
            updated_model = get_model(new_model_path)

        st.success(f"Retraining complete! New model saved at `{new_model_path}`.")


# ----------------- TAB 3: INSIGHTS -----------------

with tab_insights:
    st.subheader(" Dataset & Prediction Insights")

    st.write(
        """
        Below are some simple insights based on the model's classes.
        In your video / report you can explain:
        - How the model behaves on different classes
        - Which classes might be harder/easier
        - Any class imbalance in your training data (if you add your own images)
        """
    )

    # Simple static distribution for demonstration purposes
    st.write("### Example: Class Index vs. Example Probability Distribution")

    example_probs = np.linspace(0.05, 0.15, num=len(CLASS_NAMES))
    example_probs = example_probs / example_probs.sum()  # normalize

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.bar(CLASS_NAMES, example_probs)
    ax2.set_ylabel("Relative Importance (example)")
    ax2.set_title("Example distribution across classes")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.info(
        """
        You can later replace this with real insights, e.g.:
        - Number of images per class in `data/train/`
        - Average prediction confidence per class on a test batch
        """
    )
