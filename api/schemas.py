from pydantic import BaseModel
from typing import Dict

class FraudRequest(BaseModel):
    features: Dict[str, float]

