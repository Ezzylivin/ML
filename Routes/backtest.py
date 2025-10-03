# File: routers/backtest.py
from fastapi import APIRouter, Depends, HTTPException
from services import backtest_service # Corrected import path
from auth import get_current_user # ✅ FIX: Import from the new auth.py file
import pandas as pd
import os

router = APIRouter()

@router.post("/backtest/run")
# ✅ FIX: The 'user' variable will now contain the decoded JWT payload (e.g., {'id': '...'})
def run_backtest_endpoint(user: dict = Depends(get_current_user)):
    user_id = user.get("id")
    user_file = f"user_data/{user_id}/features_with_predictions.csv"

    if not os.path.exists(user_file):
        raise HTTPException(status_code=404, detail="No prediction data found for user.")

    df = pd.read_csv(user_file, index_col=0, parse_dates=True)
    if df.empty:
        raise HTTPException(status_code=400, detail="Data file is empty.")

    # Assuming run_backtest is a function in backtest_service
    results = backtest_service.run_backtest(df)

    return {
        "status": "ok",
        "user_id": user_id,
        "backtest_results": results
    }
