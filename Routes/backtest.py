# routers/backtest.py
from fastapi import APIRouter, Depends, HTTPException
from services.backtest_service import run_backtest
from core.deps import get_current_user
import pandas as pd
import os

router = APIRouter()

@router.post("/backtest/run")
def run_backtest_endpoint(user=Depends(get_current_user)):
    """
    Run a backtest for the current user.
    Loads their prepared features+predictions file,
    then runs strategy + ML diagnostics.
    """
    user_file = f"user_data/{user.id}/features_with_predictions.csv"

    if not os.path.exists(user_file):
        raise HTTPException(
            status_code=404,
            detail="No features_with_predictions.csv found. Please generate predictions first."
        )

    # Load with datetime index
    df = pd.read_csv(user_file, index_col=0, parse_dates=True)

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Data file is empty. Ensure predictions were saved correctly."
        )

    results = run_backtest(df)

    return {
        "status": "ok",
        "user_id": user.id,
        "backtest_results": results
    }
