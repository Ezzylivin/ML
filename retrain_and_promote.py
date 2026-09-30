#!/usr/bin/env python3
"""
retrain_and_promote.py — scheduled, self-contained walk-forward retrainer.

WHAT IT DOES (per symbol/timeframe):
  1. Refresh the market-data CSV with the latest candles from Coinbase.
  2. Rebuild features + forward-looking TP/SL labels (your existing pipeline).
  3. Train FRESH XGBoost + RandomForest experts into a STAGING folder,
     using the same regularized hyper-params as engineer_and_train.
  4. VALIDATION GATE: on the most recent holdout slice, compare each fresh
     model against the currently-live model. Promote only if the fresh model
     is at least as accurate (and clears a 0.50 baseline).
  5. On promotion: atomically swap staged files over the live files and touch
     app/models/.reload so the running engine hot-reloads on its next tick.

SCOPE: XGBoost + RandomForest (fast, CPU-only). The transformer and stacking
judge keep loading as-is; retrain them with your existing scripts
(trainTransformers.py / train_judge.py) when you want — this script is safe to
run hourly/daily without a GPU.

USAGE:
  python retrain_and_promote.py                 # BTC-USD 1h, retrain+promote
  python retrain_and_promote.py --dry-run       # train+validate, DO NOT promote
  python retrain_and_promote.py --symbols BTC-USD,ETH-USD --timeframe 1h

Cron example (every 6h):  0 */6 * * *  cd /root/Project/ML && /usr/bin/python3 retrain_and_promote.py >> logs/retrain.log 2>&1
"""

import os
import sys
import time
import argparse
import logging
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config2 import DATA_DIR, MODEL_DIR, FEATURE_COLUMNS
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels

import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [retrain] %(levelname)s %(message)s")
log = logging.getLogger("retrain")

STAGING_DIR    = os.path.join(MODEL_DIR, "_staging")
RELOAD_SENTINEL = os.path.join(MODEL_DIR, ".reload")

# Promote only if the fresh model is within this tolerance of (or better than)
# the live model on the recent holdout. 0.0 = must be >= live. A tiny negative
# tolerance avoids needless churn from run-to-run noise.
PROMOTE_TOLERANCE = -0.01
MIN_BASELINE_ACC  = 0.50
LOOK_FORWARD      = 24     # must match engineer_and_train labels
MIN_ROWS          = 600


# ----------------------------------------------------------------------------- data
def refresh_csv(symbol: str, timeframe: str) -> str:
    """Append the latest Coinbase candles to data/{SYMBOL}-{tf}.csv. Returns path."""
    import ccxt  # sync client; no event loop needed in a batch job
    path = os.path.join(DATA_DIR, f"{symbol}-{timeframe}.csv")
    existing = None
    since = None
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            tcol = next((c for c in existing.columns if c.lower() in
                         ("timestamp", "time", "date")), None)
            if tcol is not None:
                last = pd.to_datetime(existing[tcol], utc=True, errors="coerce").max()
                if pd.notna(last):
                    since = int(last.timestamp() * 1000) + 1
        except Exception as e:
            log.warning(f"could not read existing csv ({e}); refetching fresh")
            existing = None

    ex = ccxt.coinbase({"enableRateLimit": True})
    fetch_symbol = symbol.replace("-", "/")
    now_ms = int(time.time() * 1000)
    if since is None:
        # ~2 years of hourly candles as a cold-start floor
        since = now_ms - 730 * 24 * 3600 * 1000
    rows = []
    while since < now_ms:
        batch = ex.fetch_ohlcv(fetch_symbol, timeframe, since=since, limit=300)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 300:
            break

    if rows:
        new_df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        if existing is not None and len(existing):
            tcol = next((c for c in existing.columns if c.lower() in
                         ("timestamp", "time", "date")), None)
            if tcol:
                existing = existing.rename(columns={tcol: "timestamp"})
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
                combined = pd.concat([existing[["timestamp", "open", "high", "low", "close", "volume"]], new_df])
            else:
                combined = new_df
        else:
            combined = new_df
        combined = (combined.dropna(subset=["timestamp"])
                            .drop_duplicates(subset=["timestamp"])
                            .sort_values("timestamp"))
        os.makedirs(DATA_DIR, exist_ok=True)
        combined.to_csv(path, index=False)
        log.info(f"{symbol}: data refreshed -> {len(combined)} rows ({path})")
    else:
        log.info(f"{symbol}: no new candles; using existing csv")
    return path


# ----------------------------------------------------------------------------- train
def _build_xy(path: str):
    raw = pd.read_csv(path)
    df, feats = apply_mega_features(raw)
    if len(df) < MIN_ROWS:
        raise ValueError(f"only {len(df)} rows after features (need >= {MIN_ROWS})")
    df["target"] = create_strategic_labels(df, look_forward=LOOK_FORWARD, tp=1.0, sl=1.0)
    X = df[feats].values[:-LOOK_FORWARD]
    y = df["target"].values[:-LOOK_FORWARD]
    split = int(len(X) * 0.80)
    return X[:split], X[split:], y[:split], y[split:], feats


def _fresh_models():
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                        base_score=0.5, min_child_weight=10, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=1.0)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=20,
                                min_samples_leaf=10, max_features="sqrt")
    return {"xgboost": xgb, "randomforest": rf}


def _live_accuracy(model_type, ticker, timeframe, X_test, y_test, feats):
    """Accuracy of the currently-live model on this holdout, or None if absent/incompatible."""
    live_path = os.path.join(MODEL_DIR, f"{ticker}_{timeframe}_{model_type}_model.joblib")
    if not os.path.exists(live_path):
        return None
    try:
        payload = joblib.load(live_path)
        model = payload.get("model") if isinstance(payload, dict) else payload
        names = payload.get("feature_names", feats) if isinstance(payload, dict) else feats
        if list(names) != list(feats):
            # feature contract changed; treat as not comparable
            return None
        return float((model.predict(X_test) == y_test).mean())
    except Exception as e:
        log.warning(f"live {model_type} eval failed: {e}")
        return None


def retrain_symbol(symbol: str, timeframe: str, dry_run: bool) -> dict:
    ticker = symbol.split("-")[0].lower()
    log.info(f"=== {symbol} ({timeframe}) ===")
    path = refresh_csv(symbol, timeframe)
    X_tr, X_te, y_tr, y_te, feats = _build_xy(path)
    log.info(f"{symbol}: {len(X_tr)} train / {len(X_te)} test | "
             f"test positive {y_te.mean()*100:.1f}%")

    os.makedirs(STAGING_DIR, exist_ok=True)
    results = {}
    for mtype, model in _fresh_models().items():
        model.fit(X_tr, y_tr)
        new_acc  = float((model.predict(X_te) == y_te).mean())
        live_acc = _live_accuracy(mtype, ticker, timeframe, X_te, y_te, feats)

        promote = (new_acc >= MIN_BASELINE_ACC and
                   (live_acc is None or new_acc >= live_acc + PROMOTE_TOLERANCE))
        staged = os.path.join(STAGING_DIR, f"{ticker}_{timeframe}_{mtype}_model.joblib")
        joblib.dump({"model": model, "feature_names": feats}, staged)

        verdict = "PROMOTE" if promote else "KEEP-OLD"
        live_str = f"{live_acc*100:.1f}%" if live_acc is not None else "n/a"
        log.info(f"{symbol} {mtype}: new {new_acc*100:.1f}% vs live {live_str} -> "
                 f"{verdict}{' (dry-run)' if dry_run else ''}")

        if promote and not dry_run:
            live = os.path.join(MODEL_DIR, f"{ticker}_{timeframe}_{mtype}_model.joblib")
            shutil.move(staged, live)   # atomic on same filesystem
        results[mtype] = {"new_acc": new_acc, "live_acc": live_acc, "promoted": promote and not dry_run}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC-USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    any_promoted = False
    for sym in symbols:
        try:
            res = retrain_symbol(sym, args.timeframe, args.dry_run)
            any_promoted = any_promoted or any(r["promoted"] for r in res.values())
        except Exception as e:
            log.error(f"{sym}: retrain FAILED: {e}")

    if any_promoted:
        try:
            with open(RELOAD_SENTINEL, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
            log.info(f"models promoted -> touched reload sentinel {RELOAD_SENTINEL}")
        except Exception as e:
            log.warning(f"could not write reload sentinel: {e}")
    else:
        log.info("no promotions this run")


if __name__ == "__main__":
    main()
