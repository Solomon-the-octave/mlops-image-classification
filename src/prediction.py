from fastapi import FastAPI, UploadFile, File
from src.preprocessing import preprocess_image
from src.model import load_model, predict_single_image
import uvicorn

app = FastAPI()

# Load model once on startup
model = load_model()

@app.get("/")
def home():
    return {"message": "Image Classification API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(contents)

        # Preprocess
        img = preprocess_image(temp_file)

        # Predict
        label, confidence = predict_single_image(model, img)

        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
