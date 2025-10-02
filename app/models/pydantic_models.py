from pydantic import BaseModel
from typing import Optional, List

# Example request for training a model
class TrainRequest(BaseModel):
    symbol: str
    candles: int = 500
    profit_target_pct: float = 1.0
    stop_loss_pct: float = 1.0

# Example response for training
class TrainResponse(BaseModel):
    message: str
    model_path: Optional[str] = None

# Example request for prediction
class PredictRequest(BaseModel):
    symbol: str
    features: List[float]

# Example response for prediction
class PredictResponse(BaseModel):
    prediction: int
    confidence: float
