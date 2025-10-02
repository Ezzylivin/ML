from fastapi import APIRouter, HTTPException
from app.models.pydantic_models import PredictRequest, PredictResponse
from app.services import predict_service

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def make_prediction(req: PredictRequest):
    try:
        result = predict_service.predict(req.symbol, req.features)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
