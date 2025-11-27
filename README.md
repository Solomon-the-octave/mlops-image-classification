# CIFAR-10 Image Classification – MLOps Final Project

This project demonstrates a full end-to-end **MLOps workflow** using the CIFAR-10 dataset.  
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

#  **Project Links**

###  **GitHub Repository**
https://github.com/Solomon-the-octave/mlops-image-classification

###  **Live Deployed App (Streamlit)**
https://mlops-image-classification-6paedubted5c9hbaja5ydz.streamlit.app/

### **Video Demo (YouTube / Google Drive)**
https://youtu.be/ezer2JBtxwc 

---

#  **Project Description**

This MLOps project implements an image classification system using the **CIFAR-10 dataset**. The dataset contains:

- 60,000 RGB images  
- Size: 32 × 32 pixels  
- 10 categories (Cat, Dog, Airplane, Truck, Car, etc.)

The workflow includes:

### **1. Model Training**
A Convolutional Neural Network (CNN) with:
- 3 convolutional blocks  
- Batch normalization  
- MaxPooling  
- Dense layer with dropout  
- Softmax output layer  
- Data augmentation for robust learning  
- EarlyStopping callback  

### **2. Evaluation Metrics**
The notebook evaluates the model using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Classification report  
- Confusion matrix  
- Training and validation curves  

### **3. Model Versioning**
The trained model is exported as:
models/base_cifar10_model.h5
models/retrained_cifar10_model.h5


### **4. Retraining Pipeline**
The notebook simulates new incoming data:
- Creates `new_data.npz`
- Uploads new data via Colab  
- Normalizes the data  
- Retrains the existing model (fine-tuning)  
- Saves a new version of the model

### **5. Deployment**
A Streamlit app is deployed using:
- `app.py`
- `base_cifar10_model.h5`
- Custom prediction function  
- Full UI for uploading images and getting predictions  

Users can upload any image and receive:
- Predicted class  
- Confidence score  
- Class probability breakdown  

---

#  **How to Run This Project Locally**

## **1. Clone the repository**
```bash
git clone https://github.com/Solomon-the-octave/mlops-image-classification
cd mlops-image-classification

Install dependencies
pip install -r requirements.txt

Run the Streamlit app
streamlit run app.py
 Retraining the Model (Local or Colab)
Step 1 — Prepare new_data.npz

You can simulate new data using:

x_new = x_train_full[:500]
y_new = y_train_full[:500]
np.savez("new_data.npz", x_new=x_new, y_new=y_new)

Step 2 — Upload in Colab
from google.colab import files
uploaded = files.upload()

Step 3 — Preprocess and Retrain
data = np.load("new_data.npz")
x_new = data["x_new"].astype("float32") / 255.0
y_new = data["y_new"]

model.fit(x_new, y_new, epochs=2, validation_data=(x_val, y_val))

Step 4 —

Save retrained model:

model.save("models/retrained_cifar10_model.h5")

 Load Testing (Flood Request Simulation)

To evaluate model performance under load, Locust was used.

Locust Command
locust -f locustfile.py --host http://0.0.0.0:8000 --headless -u 20 -r 5 -t 30s

What Was Tested

Response time of the prediction endpoint

Latency under increasing request traffic

Throughput (requests per second)

Load Test Result Summary

Model inference remained stable under simulated traffic

FastAPI prediction latency remained within expected bounds

No server errors or timeouts were observed

System handled up to 20 users with ramp-up of 5 users/second

This validates that the model and API can support real-world usage.

 Notebook Contents (mlops_image.ipynb)

The notebook includes:

✔️ Data Loading & Preprocessing
✔️ Train/Val/Test Split
✔️ Data Augmentation Pipeline
✔️ Model Architecture Definition
✔️ Training with EarlyStopping
✔️ Accuracy, Precision, Recall, F1-score
✔️ Confusion Matrix
✔️ Prediction Function
✔️ Model Saving
✔️ new_data.npz Creation
✔️ Upload + Preprocessing
✔️ Retraining + Versioning
