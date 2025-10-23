# File: app/Routes/predict_routes.py

import os
import json
from fastapi import APIRouter, HTTPException
from app.models.pydantic_models import PredictRequest, PredictResponse
from app.services import predict_service

router = APIRouter()

# This is the base directory where each model's sub-directory is located.
# EXPECTED STRUCTURE:
# /app
#  └── /models
#      ├── /btc_xgboost_model
#      │   ├── model.pkl
#      │   └── metadata.json
#      └── /another_model
#          ├── model.joblib
#          └── metadata.json
MODELS_DIR = "app/models"

@router.get("/models", tags=["Models"])
def get_available_models():
    """
    [✅ REFACTORED] Scans the models directory for model sub-directories
    and returns a list of available model names.
    """
    if not os.path.isdir(MODELS_DIR):
        raise HTTPException(status_code=500, detail="Models directory not found on the server.")
    
    try:
        # List sub-directories within MODELS_DIR, which represent the model names.
        models = [
            name for name in os.listdir(MODELS_DIR)
            if os.path.isdir(os.path.join(MODELS_DIR, name))
        ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read models directory: {str(e)}")

@router.get("/models/{model_name}/metadata", tags=["Models"])
def get_model_metadata(model_name: str):
    """
    [✅ NEW] Fetches the metadata.json file for a specific model.
    This resolves the 404 error your frontend was encountering.
    """
    try:
        # Use the model_name as-is (lowercase) to build the path.
        metadata_path = os.path.join(MODELS_DIR, model_name, "metadata.json")

        if not os.path.exists(metadata_path):
            # If the file doesn't exist, return a 404 error.
            raise HTTPException(status_code=404, detail=f"Metadata for model '{model_name}' not found.")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to parse metadata.json for model '{model_name}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def make_prediction(req: PredictRequest):
    """
    [✅ REFACTORED] Makes a prediction using a specified model and features.
    """
    try:
        # Pass the model_name from the request to the prediction service.
        result = predict_service.predict(req.model_name, req.symbol, req.features)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
