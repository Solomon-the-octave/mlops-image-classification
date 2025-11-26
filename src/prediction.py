from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil

from preprocessing import preprocess_image
from model import (
    load_model,
    predict_single_image,
    retrain_model,
    BASE_MODEL_PATH
)

# ---------- FastAPI app setup ----------

app = FastAPI(
    title="MLOps Image Classification API",
    description="API for image prediction and model retraining (CIFAR-10 style).",
    version="1.0.0"
)

# Allow all origins for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup
model = load_model(BASE_MODEL_PATH)


# ---------- Helper functions ----------

def save_uploaded_file(upload_file: UploadFile, dest_path: str):
    """
    Save an uploaded file to the destination path.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


# ---------- API endpoints ----------

@app.get("/")
def root():
    return {
        "message": "Image Classification API is up.",
        "endpoints": ["/predict", "/upload_data", "/retrain"]
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class of a single uploaded image.

    Usage:
        - Send a POST request with form-data:
            key: "file", value: (image file)
    """
    global model

    # Preprocess image
    img_array = preprocess_image(file.file)

    # Make prediction
    predicted_class, confidence, probs = predict_single_image(model, img_array)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs.tolist()
    }


@app.post("/upload_data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_name: str = Form(...)
):
    """
    Upload multiple images to be used for retraining.

    The images will be saved under:
        data/train/<class_name>/

    Usage (from UI or tools like Postman):
        - form-data:
            - key: "files" (multiple file uploads)
            - key: "class_name" (text) e.g. "cat"
    """
    save_dir = os.path.join("data", "train", class_name)

    sav

