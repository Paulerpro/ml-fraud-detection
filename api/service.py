import torch
import joblib
import pandas as pd
from ml_fraud_detection.training.model import MLP
from typing import Dict
import mlflow

scaler = joblib.load("models/scaler.pkl")

model = MLP(input_dim=30) 

# model_name = "FraudDetectionModel"
# model_uri = f"models:/{model_name}/latest"
# model = mlflow.pytorch.load_model(model_uri)

model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device("cpu")))
model.eval()

def predict_fraud(features: Dict[str, float]):
    """Infer from model & scaler"""

    X = pd.DataFrame([features])

    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        pred = torch.sigmoid(logits).item()

    return {
        "fraud_probability": pred,
        "is_fraud": bool(pred >= 0.5)
    }
