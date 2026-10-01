#!/usr/bin/env python3
"""
walkforward_backtest.py — unbiased, out-of-sample edge test for the experts.

WHY NOT just backtest the saved models: the saved council (xgboost, randomforest,
transformer, judge) was trained on the ENTIRE price history, so testing it on any
past window leaks the future into the model — results look great and mean nothing.

WHAT THIS DOES INSTEAD: rolling walk-forward. It splits history into K sequential
folds; for each fold it trains fresh XGBoost + RandomForest on ALL data BEFORE the
fold and evaluates on the fold it has never seen. That is the honest question:
"trained only on the past, does the model predict the next chunk better than a coin
flip?" It reports per-fold out-of-sample accuracy and a directional-return proxy,
plus an aggregate verdict.

This tests the EXPERTS' predictive signal (the core edge). It does not simulate the
full live strategy — the real forward truth is the paper-trading ledger
(ledger_report.py). Use both.

Usage:
  python walkforward_backtest.py --symbol BTC-USD --timeframe 1h --folds 6
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from app.config2 import DATA_DIR
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

LOOK_FORWARD = 24


def _fresh():
    return {
        "xgboost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                 base_score=0.5, min_child_weight=10, subsample=0.8,
                                 colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=1.0),
        "randomforest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=20,
                                               min_samples_leaf=10, max_features="sqrt"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    path = os.path.join(DATA_DIR, f"{args.symbol}-{args.timeframe}.csv")
    if not os.path.exists(path):
        print(f"No data at {path}. Fetch history first (retrain_and_promote refreshes it).")
        return

    df, feats = apply_mega_features(pd.read_csv(path))
    df["target"] = create_strategic_labels(df, look_forward=LOOK_FORWARD, tp=1.0, sl=1.0)
    df = df.iloc[:-LOOK_FORWARD]  # labels near the end are unreliable
    X = df[feats].values
    y = df["target"].values
    ret = df["close"].pct_change().shift(-1).fillna(0).values  # next-bar return proxy

    n = len(X)
    if n < 800:
        print(f"Only {n} rows — need more history for a meaningful walk-forward.")
        return

    # Sequential folds over the back half; first half is the initial training base.
    start = n // 2
    fold_size = (n - start) // args.folds
    print("=" * 70)
    print(f"  WALK-FORWARD (out-of-sample)  {args.symbol} {args.timeframe}  | {n} bars, {args.folds} folds")
    print("=" * 70)
    print(f"  {'fold':<5}{'test bars':<11}{'xgb acc':<10}{'rf acc':<10}{'ens acc':<10}{'ret/trade':<10}")
    print("-" * 70)

    accs = []
    rets = []
    for i in range(args.folds):
        te0 = start + i * fold_size
        te1 = te0 + fold_size if i < args.folds - 1 else n
        if te1 - te0 < 20:
            continue
        Xtr, ytr = X[:te0], y[:te0]
        Xte, yte = X[te0:te1], y[te0:te1]
        rte = ret[te0:te1]

        preds = {}
        for name, m in _fresh().items():
            m.fit(Xtr, ytr)
            preds[name] = m.predict_proba(Xte)[:, 1]
        xgb_acc = ((preds["xgboost"] > 0.5).astype(int) == yte).mean()
        rf_acc = ((preds["randomforest"] > 0.5).astype(int) == yte).mean()
        ens = (preds["xgboost"] + preds["randomforest"]) / 2
        ens_pred = (ens > 0.5).astype(int)
        ens_acc = (ens_pred == yte).mean()
        # directional return proxy: long when ens>0.5 else flat-short
        direction = np.where(ens > 0.5, 1, -1)
        ret_per = float((direction * rte).mean())

        accs.append(ens_acc)
        rets.append(ret_per)
        print(f"  {i:<5}{te1-te0:<11}{xgb_acc*100:<10.1f}{rf_acc*100:<10.1f}{ens_acc*100:<10.1f}{ret_per*100:<10.3f}")

    print("-" * 70)
    if accs:
        mean_acc = np.mean(accs) * 100
        pos_folds = sum(1 for r in rets if r > 0)
        mean_ret = np.mean(rets) * 100
        print(f"  Mean OOS ensemble accuracy: {mean_acc:.1f}%   (50% = coin flip)")
        print(f"  Folds with positive return proxy: {pos_folds}/{len(rets)}")
        print(f"  Mean return proxy / bar: {mean_ret:.3f}%  (before fees/slippage)")
        edge = mean_acc > 52.0 and pos_folds >= (len(rets) * 0.6)
        print(f"  VERDICT: {'SIGNS OF EDGE — validate in paper next' if edge else 'NO RELIABLE OOS EDGE in the experts'}")
        print("  NOTE: accuracy a few points above 50% is normal and fragile; the")
        print("  return proxy ignores fees/slippage, so real net edge is lower. The")
        print("  paper ledger is the final word.")
    print("=" * 70)


if __name__ == "__main__":
    main()
