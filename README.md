
# Credit Card Fraud Detection API

This repository hosts the backend for a credit card fraud detection system. It includes:

- Data preprocessing and training pipeline
- Model serialization (Random Forest/XGBoost)
- A FastAPI-based REST API
- Render deployment support
- Integration-ready with Streamlit frontend
## 🚀 Live API

**Base URL:**  
[https://credit-card-fraud-detection-api.onrender.com](https://credit-card-fraud-detection-api.onrender.com)

You can test it live via the built-in Swagger UI at:  
`https://credit-card-fraud-detection-api.onrender.com/docs`


## 🧠 How it Works

1. **Model Training** (`train_model.py`):
   - Loads credit card transaction data
   - Handles preprocessing (scaling, encoding, SMOTE)
   - Trains Random Forest or XGBoost with GridSearchCV
   - Saves `fraud_model.pkl`, `scaler.pkl`, and `features.pkl` to the `/models` folder

2. **API Serving** (`main.py`):
   - Loads trained model and preprocessors
   - Accepts POST requests at `/predict`
   - Returns whether the transaction is fraudulent with a probability score

## 🗂️ Project Structure

This repository includes all components required to train, deploy, and test a credit card fraud detection model using FastAPI and optionally Streamlit. The models folder contains the trained model (fraud_model.pkl), the scaler used for feature normalization (scaler.pkl), and the feature list (features.pkl) required for making predictions.

The main.py file is the core FastAPI application that serves the prediction API. train_model.py is the script used to load the dataset, preprocess the data, train the model, evaluate it, and save the necessary model artifacts.

The streamlit_app.py file provides a simple web interface built using Streamlit that allows users to input transaction data and view predictions from the deployed API.

requirements.txt lists all the Python dependencies needed to run the project. start.sh is a shell script that runs the FastAPI server and is used during deployment on Render. The render.yaml file contains the deployment configuration specific to the Render platform. .gitignore ensures that sensitive or unnecessary files like virtual environments and temporary data are excluded from version control. A LICENSE file is also included to define the usage rights for this project.



## 🧪 Example Input (JSON)

Send a `POST` request to `/predict` with the following format:

```json
{
  "Time": 15000,
  "V1": -1.3598,
  "V2": -0.0728,
  "V3": 2.5363,
  "V4": 1.3781,
  "V5": -0.3383,
  "V6": 0.4623,
  "V7": 0.2395,
  "V8": 0.0986,
  "V9": 0.3637,
  "V10": 0.0907,
  "V11": -0.5515,
  "V12": -0.6178,
  "V13": -0.9913,
  "V14": -0.3111,
  "V15": 1.4681,
  "V16": -0.4704,
  "V17": 0.2079,
  "V18": 0.0257,
  "V19": 0.4039,
  "V20": 0.2514,
  "V21": -0.0183,
  "V22": 0.2778,
  "V23": -0.1104,
  "V24": 0.0669,
  "V25": 0.1285,
  "V26": -0.1891,
  "V27": 0.1335,
  "V28": -0.0210,
  "Amount": 149.62
}
```
## Sample Output

{
 
  "prediction": 0,

  "probability": 0.014729,

  "result": "Not Fraud"

}

## Set Up Locally

### 1) Clone the Repo
git clone https://github.com/your-username/credit-card-fraud-detection-api.git
cd credit-card-fraud-detection-api

### 2) Install Dependencies
pip install -r requirements.txt

### 3) Create an env. file
DATA_PATH=archive/creditcard.csv

MODEL_DIR=models

FAST_MODE=True

### 4) Train the model
python train_model.py

### 5) Run the Fast API 
uvicorn main:app --reload

Visit http://localhost:8000/docs to interact with the API.




## Deployment (Render) 

This app is deployed to Render using:

render.yaml for service configuration

start.sh to launch the app with Uvicorn
## Dataset

Model is trained on the public Kaggle Credit Card Fraud Detection Dataset.


## Author
Shubhashish Garimella

Made using FastAPI, Scikit-learn and XGBoost
