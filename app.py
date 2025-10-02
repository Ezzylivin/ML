from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from app.services.feature_engineering import engineer_features

app = FastAPI(
    title="TradingBot ML Service",
    description="API for feature engineering and ML model support",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok", "message": "ML Feature Engineering Service running 🚀"}

@app.post("/engineer/")
async def engineer(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file, parse_dates=True, index_col=0)

        # Run your custom feature engineering
        features = engineer_features(df)

        # Optionally save the engineered features to disk
        # features.to_csv("app/data/engineered_features.csv")

        return {
            "rows": int(features.shape[0]),
            "columns": features.columns.tolist(),
            "sample": features.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")
