from fastapi import FastAPI, UploadFile, File
import pandas as pd
from feature_engineering import engineer_features

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "ML Feature Engineering Service running!"}

@app.post("/engineer/")
async def engineer(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, parse_dates=True)
    features = engineer_features(df)
    return {"rows": len(features), "columns": features.columns.tolist()}
