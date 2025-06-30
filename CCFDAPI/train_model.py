from dotenv import load_dotenv
import os

load_dotenv()

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Constants
DATA_PATH = os.getenv("DATA_PATH", r"C:\Users\shubh\Downloads\archive\creditcard.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
FAST_MODE = os.getenv("FAST_MODE", "True") == "True"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(local_path=None):
    if local_path and os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        raise FileNotFoundError("Please provide a valid dataset path.")

def detect_target_column(df):
    for target in ["Class", "is_fraud", "fraud"]:
        if target in df.columns:
            return target
    raise ValueError("No recognized target column found.")

def preprocess_data(df):
    target_col = detect_target_column(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    scaler = StandardScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    encoder = None
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = X.drop(columns=cat_cols).reset_index(drop=True)
        X = pd.concat([X, encoded_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    joblib.dump(scaler, SCALER_PATH)
    if encoder:
        joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=10, n_jobs=-1)
    }

    param_grids = {
        "Random Forest": {
            'n_estimators': [50 if FAST_MODE else 100],
            'max_depth': [None] if FAST_MODE else [None, 10],
            'min_samples_split': [2],
        },
        "XGBoost": {
            'n_estimators': [50 if FAST_MODE else 100],
            'max_depth': [3] if FAST_MODE else [3, 5],
            'learning_rate': [0.1],
        }
    }

    best_model = None
    best_f1 = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(model, param_grids[name], scoring='f1', cv=2, verbose=0, n_jobs=-1)
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        f1 = f1_score(y_test, preds)
        print(f"{name} F1 Score: {f1:.4f}")
        print("Classification Report:\n", classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("ROC AUC: {:.4f}".format(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])))

        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with F1 Score = {best_f1:.4f}")
    return best_model

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved at: {path}")

if __name__ == "__main__":
    print("Training model...")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_model(best_model)
