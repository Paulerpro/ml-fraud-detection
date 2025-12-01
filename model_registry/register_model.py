
import torch
import pandas as pd
import mlflow
import mlflow.pytorch
import dagshub

from ml_fraud_detection.training.model import MLP 

dagshub.init(
    repo_owner='paulerpro',
    repo_name='ml-fraud-detection',
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/paulerpro/ml-fraud-detection.mlflow")

run_id = "538c7bbd42bd7522ec94c3a21f7ef1728afa0030" 

model_uri = f"runs:/{run_id}/model"

registered_model_name = "fraudModel"  

result = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(f"Model registered! Name: {registered_model_name}, Version: {result.version}")

# INPUT_DIM = 30  
# model = MLP(input_dim=INPUT_DIM)

# state_dict = torch.load("models/best_model.pt", map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()  

# example_input = pd.DataFrame([{
#     "Time": 0.0,
#     "Amount": 0.0,
#     **{f"V{i}": 0.0 for i in range(1, 29)}
# }])

# with mlflow.start_run(run_name="register_best_model"):

#     mlflow.pytorch.log_model(
#         pytorch_model=model,
#         artifact_path="model",
#         input_example=example_input
#     )

#     mlflow.log_artifact("models/scaler.pkl", artifact_path="preprocessing")

#     mlflow.log_param("input_dim", INPUT_DIM)

# print("Model and scaler successfully registered in DagsHub MLflow!")

