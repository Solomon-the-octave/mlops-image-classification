 CIFAR-10 Image Classification – MLOps Demo

This project is an end-to-end MLOps workflow for an image classification model trained on the CIFAR-10 dataset.
It includes model training, evaluation, saving, and a fully deployed Streamlit web app for real-time image prediction.

The goal of this project is to demonstrate a complete machine-learning pipeline:
data → model training → metrics → saving the model → deployment → live predictions.

 Live Demo (Deployed on Streamlit)

 Try the app here:
 
https://mlops-image-classification-6paedubted5c9hbaja5ydz.streamlit.app/

You can upload an image (PNG/JPG) and the model will classify it as one of the 10 CIFAR-10 classes:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

 About the Model

The model is a Convolutional Neural Network (CNN) trained using TensorFlow/Keras.
Key features include:

Data preprocessing and normalization

Data augmentation (flip, rotation, zoom)

3 convolutional blocks with BatchNorm + MaxPooling

Dense classification head with Dropout

Trained with EarlyStopping

Exported as base_cifar10_model.h5

The model achieves solid accuracy on CIFAR-10 while staying lightweight enough for fast inference.

 Project Structure
mlops-image-classification/
│
├── app.py                 # Streamlit prediction app
├── requirements.txt       # Packages needed for deployment
│
├── models/
│   └── base_cifar10_model.h5   # Saved trained model
│
├── notebook/
│   └── mlops_image.ipynb       # Full training + evaluation notebook
│
├── src/
│   ├── preprocess_image.py     # Image preprocessing utilities
│   └── prediction.py           # Prediction helper functions
│
└── README.md              # Project documentation

 Technologies Used

Python

TensorFlow / Keras

NumPy & Matplotlib

scikit-learn

Streamlit

GitHub for version control

Streamlit Cloud for deployment

 How the MLOps Workflow Works
1️ Model Training (Colab Notebook)

Loaded CIFAR-10 dataset

Split into train/val/test

Applied data augmentation

Trained a CNN model

Evaluated with accuracy, precision, recall, F1-score

Saved model to /models/base_cifar10_model.h5

2️ Building the Streamlit App

Loads the saved model

Preprocesses user-uploaded images (resize → normalize → batch)

Runs prediction

Displays:

Predicted class

Confidence score

Full probability distribution

3️ Deployment

The repo was connected to Streamlit Cloud, which automatically:

Installs dependencies from requirements.txt

Runs app.py

Hosts a live, public demo with a clean UI

 Sample Prediction

Upload any image, and the app will output:

Prediction: automobile (confidence 0.82)


With detailed class probabilities below it.

 Why This Project Matters

This project demonstrates a complete, beginner-friendly MLOps pipeline — not just model training, but real deployment.
It shows how to take a model from a notebook into a production-style application that anyone can use.

Author

Solomon-the-octave
Built for an academic MLOps summative assessment.

If you’d like improvements or new features (better UI, CAM heatmaps, improved model, etc.), feel free to contribute or reach out.
