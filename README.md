# CIFAR-10 Image Classification – MLOps Final Project

This project demonstrates a full end-to-end MLOps workflow using the CIFAR-10 dataset.  
It includes:

- Data preprocessing  
- CNN model training  
- Evaluation using multiple metrics  
- Model versioning  
- Retraining with new uploaded data  
- Deployment on Streamlit for real-time prediction  
- Load testing using Locust  
- A complete MLOps pipeline structure  

---

# Project Links

GitHub Repository  
https://github.com/Solomon-the-octave/mlops-image-classification

Live Deployed App (Streamlit)  
https://mlops-image-classification-6paedubted5c9hbaja5ydz.streamlit.app/

Video Demo (YouTube / Google Drive)  
https://youtu.be/ezer2JBtxwc

---

# Project Description

This MLOps project implements a full machine learning operations workflow for image classification on the CIFAR-10 dataset.  
The dataset contains 60,000 RGB images sized 32x32 pixels across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The workflow covers data preprocessing, neural network training, evaluation, retraining, versioning, cloud deployment, and load testing.

---

## 1. Model Training

The project uses a Convolutional Neural Network (CNN) with:

- Three convolutional blocks  
- Batch normalization  
- MaxPooling layers  
- Dense layer with dropout  
- Softmax output for multiclass classification  
- Data augmentation  
- EarlyStopping for regularization  

---

## 2. Evaluation Metrics

Model evaluation includes:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Classification report  
- Confusion matrix  
- Training and validation learning curves  

These results are presented in the notebook.

---

## 3. Model Versioning

Two model files are provided:

models/base_cifar10_model.h5
models/retrained_cifar10_model.h5


The first is the original trained model, and the second is produced after retraining with new uploaded data.

---

## 4. Retraining Pipeline

The retraining workflow includes:

- Creating a new dataset file (`new_data.npz`)  
- Uploading new data in Google Colab  
- Normalizing the data  
- Fine-tuning the existing model  
- Saving a new version of the model  

This simulates a real MLOps retraining scenario.

---

## 5. Deployment

Deployment is done using Streamlit.  
The UI allows users to:

- Upload an image  
- Receive the predicted class  
- View confidence scores  
- View probability distribution across all 10 classes  

The application loads `base_cifar10_model.h5` from the repository.

---

# How to Run This Project Locally

## 1. Clone the repository
```bash
git clone https://github.com/Solomon-the-octave/mlops-image-classification
cd mlops-image-classification

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit application
streamlit run app.py


The application will open at:

http://localhost:8501

Retraining the Model (Local or Colab)
Step 1: Prepare new_data.npz
x_new = x_train_full[:500]
y_new = y_train_full[:500]
np.savez("new_data.npz", x_new=x_new, y_new=y_new)

Step 2: Upload in Colab
from google.colab import files
uploaded = files.upload()

Step 3: Preprocess and Retrain
data = np.load("new_data.npz")
x_new = data["x_new"].astype("float32") / 255.0
y_new = data["y_new"]

model.fit(x_new, y_new, epochs=2, validation_data=(x_val, y_val))

Step 4: Save retrained model
model.save("models/retrained_cifar10_model.h5")

Load Testing (Flood Request Simulation)

Load testing was conducted using Locust to evaluate API performance under stress.

Locust Command
locust -f locustfile.py --host http://0.0.0.0:8000 --headless -u 20 -r 5 -t 30s

Test Results Summary

Prediction endpoint responded consistently under load

Latency remained stable

No timeouts or server crashes occurred

System handled up to 20 simulated users with a ramp-up rate of 5 users per second

This confirms that the model API can support concurrent users.

Notebook Contents (mlops_image.ipynb)

The notebook contains:

Data loading and preprocessing

Train/validation/test split

Data augmentation

Model architecture

Training with EarlyStopping

Evaluation metrics (accuracy, precision, recall, F1)

Confusion matrix

Prediction demonstrations

Model saving

Retraining workflow

Versioning of models

Model Files
models/base_cifar10_model.h5
models/retrained_cifar10_model.h5


The first is the original trained model; the second is generated after applying the retraining workflow.

About

This project showcases an end-to-end MLOps pipeline for image classification, including training, evaluation, retraining, deployment, and load testing, demonstrating complete lifecycle management for machine learning models.
