"""
ledger_trainer.py — train a 'trade-quality' model from the bot's OWN closed
trades (data/trade_ledger.db): the entry-time feature snapshot -> realized
win/loss. This is the truest self-learning signal — the bot learns which of
*its own* setups actually worked, not just what price did.

Saved as {ticker}_{tf}_ledger_model.joblib and consumed by the live engine as an
optional extra entry gate. Fully guarded: with no ledger, too few trades, or a
one-sided history it trains nothing and the live gate is a no-op.
"""
import os
import json
import sqlite3
import logging

import numpy as np
import pandas as pd
import joblib

from app.config2 import DATA_DIR, MODEL_DIR, FEATURE_COLUMNS
from xgboost import XGBClassifier

log = logging.getLogger("ledger_trainer")

LEDGER_PATH   = os.path.join(DATA_DIR, "trade_ledger.db")
MIN_SAMPLES   = 60     # need enough trades before the signal means anything
MIN_PER_CLASS = 15     # and a reasonable number of BOTH wins and losses


def _ticker(symbol: str) -> str:
    return symbol.split("-")[0].lower()


def train_ledger_model(symbol: str, timeframe: str = "1h") -> dict:
    """Train + save a win/loss classifier from this symbol's closed trades.
    Returns a small status dict; never raises."""
    if not os.path.exists(LEDGER_PATH):
        return {"symbol": symbol, "trained": False, "reason": "no ledger yet"}
    try:
        conn = sqlite3.connect(LEDGER_PATH, timeout=10)
        df = pd.read_sql_query(
            "SELECT features_json, label FROM trades "
            "WHERE symbol = ? AND features_json IS NOT NULL",
            conn, params=(symbol,),
        )
        conn.close()
    except Exception as e:
        return {"symbol": symbol, "trained": False, "reason": f"read failed: {e}"}

    if len(df) < MIN_SAMPLES:
        return {"symbol": symbol, "trained": False,
                "reason": f"only {len(df)} trades (need >= {MIN_SAMPLES})"}

    rows = []
    for fj in df["features_json"]:
        try:
            d = json.loads(fj) or {}
        except Exception:
            d = {}
        rows.append([float(d.get(c, 0.0)) for c in FEATURE_COLUMNS])
    X = np.asarray(rows, dtype=float)
    y = df["label"].astype(int).values
    wins = int(y.sum()); losses = int(len(y) - wins)
    if wins < MIN_PER_CLASS or losses < MIN_PER_CLASS:
        return {"symbol": symbol, "trained": False,
                "reason": f"need >= {MIN_PER_CLASS} of each class (have {wins}W/{losses}L)"}

    try:
        model = XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, base_score=0.5,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, gamma=1.0,
        )
        model.fit(X, y)
        acc = float((model.predict(X) == y).mean())
        path = os.path.join(MODEL_DIR, f"{_ticker(symbol)}_{timeframe}_ledger_model.joblib")
        joblib.dump({"model": model, "feature_names": list(FEATURE_COLUMNS),
                     "n_samples": int(len(y)), "win_rate": float(y.mean())}, path)
        log.info(f"ledger model {symbol}: {len(y)} trades ({wins}W/{losses}L), "
                 f"train acc {acc*100:.1f}% -> {path}")
        return {"symbol": symbol, "trained": True, "n": int(len(y)),
                "win_rate": round(float(y.mean()) * 100, 1), "acc": round(acc, 3)}
    except Exception as e:
        return {"symbol": symbol, "trained": False, "reason": f"fit failed: {e}"}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTC-USD", "ETH-USD", "SOL-USD"]
    for s in syms:
        print(train_ledger_model(s))
