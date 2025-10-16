# File: app/Routes/predict_routes.py

import os
from fastapi import APIRouter, HTTPException
from app.models.pydantic_models import PredictRequest, PredictResponse
from app.services import predict_service

router = APIRouter()

# The path on your server where your saved model files (.pkl, .joblib) are stored.
# IMPORTANT: Make sure this path is correct for your project structure.
MODELS_DIR = "app/models/binaries" 

@router.get("/models", tags=["Models"])
def get_available_models():
    """
    Scans the models directory and returns a list of available model files.
    """
    if not os.path.isdir(MODELS_DIR):
        raise HTTPException(status_code=404, detail="Models directory not found on the server.")
    
    try:
        # This code finds all files ending in .pkl or .joblib,
        # removes the file extension, and returns the clean names.
        models = [
            os.path.splitext(f)[0] 
            for f in os.listdir(MODELS_DIR) 
            if f.endswith(('.pkl', '.joblib'))
        ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read models directory: {str(e)}")

@router.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def make_prediction(req: PredictRequest):
    """
    Makes a prediction using a specified model and features.
    """
    try:
        # Note: We'll eventually need to pass req.model_name to the service
        result = predict_service.predict(req.symbol, req.features)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
