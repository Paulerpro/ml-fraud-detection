from fastapi import FastAPI
from api.schemas import FraudRequest
from api.service import predict_fraud

app = FastAPI(title="Fraud Detection API")

@app.post("/predict")
def predict(data: FraudRequest):
    response = predict_fraud(data.features)
    return response
