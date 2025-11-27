import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_image
from src.model import (
    load_model,
    predict_single_image,
    retrain_model,
    BASE_MODEL_PATH,
    CLASS_NAMES
)

# -------------------------
# STREAMLIT CONFIG
# -------------------------

st.set_page_config(
    page_title="MLOps Image Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title(" MLOps Image Classification System")
st.write("""
This demo shows:
- Single image prediction  
- Upload + retraining  
- Basic insights  
""")

# Load base model
@st.cache_resource
def get_model():
    return load_model(BASE_MODEL_PATH)

model = get_model()

tab_predict, tab_retrain, tab_insights = st.tabs(
    [" Predict", "Retrain Model", " Insights"]
)

# -------------------------
# TAB 1 — PREDICTION
# -------------------------

with tab_predict:
    st.subheader(" Predict a Single Image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, width=200)

        img_array = preprocess_image(uploaded_file)
        pred_class, confidence, probs = predict_single_image(model, img_array)

        st.success(f"Predicted: **{pred_class}** ({confidence:.2f} confidence)")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(CLASS_NAMES)), probs)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        st.pyplot(fig)


# -------------------------
# TAB 2 — RETRAIN
# -------------------------

with tab_retrain:
    st.subheader(" Upload Training Data")

    class_name = st.text_input("Class name (folder name)", value="custom_class")

    files = st.file_uploader(
        "Upload multiple training images",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"]
    )

    if st.button(" Save Uploaded Images"):
        if not files:
            st.warning("Upload at least one image.")
        else:
            save_dir = os.path.join("data", "train", class_name)
            os.makedirs(save_dir, exist_ok=True)

            for f in files:
                with open(os.path.join(save_dir, f.name), "wb") as out:
                    out.write(f.read())

            st.success(f"Saved {len(files)} images to {save_dir}")

    st.markdown("---")

    if st.button(" Trigger Retraining"):
        with st.spinner("Retraining model..."):
            new_path = retrain_model()
            st.cache_resource.clear()     # refresh model
            model = get_model()

        st.success(f"Model retrained and saved to {new_path}")


# -------------------------
# TAB 3 — INSIGHTS
# -------------------------

with tab_insights:
    st.subheader(" Simple Data Insights")

    st.write("Example class probability distribution:")

    example_probs = np.linspace(0.05, 0.15, len(CLASS_NAMES))
    example_probs /= example_probs.sum()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(CLASS_NAMES, example_probs)
    ax.set_title("Example Class Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)
