import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.getcwd())

from app.config2 import DATA_DIR, MODEL_STORAGE_DIR, FEATURE_COLUMNS


def apply_mega_features(df):
    """Shared feature engine. Used by training, backtesting, and live engine."""
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Moving Averages
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_20'] = ta.ema(df['close'], length=20)
    
    # 2. Oscillators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df.iloc[:, 0] if adx_df is not None else 0
    
    # 3. Bollinger & Supertrend
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['BBL_20_2.0_2.0'] = bb.iloc[:, 0] if bb is not None else 0
    df['BBU_20_2.0_2.0'] = bb.iloc[:, 2] if bb is not None else 0
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['st_trend'] = st.iloc[:, 1] if st is not None else 0

    # 4. Momentum
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['STOCHk_14_3_3'] = stoch.iloc[:, 0] if stoch is not None else 50
    macd = ta.macd(df['close'])
    df['MACD_12_26_9'] = macd.iloc[:, 0] if macd is not None else 0
    df['MACDs_12_26_9'] = macd.iloc[:, 2] if macd is not None else 0

    # 5. Price Action / Vol
    df['pa_high'] = df['high'].rolling(window=20).max()
    df['pa_low'] = df['low'].rolling(window=20).min()
    df['vol_ma'] = ta.sma(df['volume'], length=20)
    
    # 6. Model Logic Features
    df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
    df['atr_logic'] = (df['atr'] / df['close']) * 1000
    df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)
    
    return df.dropna(subset=FEATURE_COLUMNS), FEATURE_COLUMNS


def create_strategic_labels(df, look_forward=24, tp=1.0, sl=1.0):
    """
    Forward-looking TP/SL labeler.
    
    For each candle, look forward N bars. If price hits TP% first -> 1 (win).
    If price hits SL% first -> 0 (loss). If neither within the window -> 0.
    
    This matches how your bot actually trades (with TP/SL targets),
    unlike the old naive "did next candle go up?" label.
    
    Args:
        df: DataFrame with 'close', 'high', 'low' columns
        look_forward: Number of bars to look ahead
        tp: Take profit percentage (1.0 = 1%)
        sl: Stop loss percentage (1.0 = 1%)
    
    Returns:
        Series of labels: 1 = TP hit first, 0 = SL hit first or neither
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = np.zeros(len(df), dtype=int)
    
    tp_mult = tp / 100.0
    sl_mult = sl / 100.0
    
    for i in range(len(df) - look_forward):
        entry = closes[i]
        tp_price = entry * (1 + tp_mult)
        sl_price = entry * (1 - sl_mult)
        
        for j in range(i + 1, min(i + look_forward + 1, len(df))):
            # Check SL first (conservative: assume worst case happens first)
            if lows[j] <= sl_price:
                labels[i] = 0
                break
            if highs[j] >= tp_price:
                labels[i] = 1
                break
    
    return pd.Series(labels, index=df.index)


def create_atr_labels(df, look_forward=24, tp_atr=3.0, sl_atr=1.5):
    """Forward TP/SL labeler aligned to the LIVE bot's ATR-based exits (long side):
    entry = close, TP = entry + tp_atr*ATR, SL = entry - sl_atr*ATR. Label = 1 if
    TP is hit before SL within look_forward bars, else 0 (SL-first or neither).

    This makes every model (experts, transformer, judge) predict what the bot
    ACTUALLY does — a 3:1.5 ATR trade — instead of a fixed symmetric 1%/1% move,
    so the model's P(win) lines up with the trade the engine really takes.
    """
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    atr    = df['atr'].values if 'atr' in df.columns else (closes * 0.01)
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - look_forward):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry    = closes[i]
        tp_price = entry + tp_atr * a
        sl_price = entry - sl_atr * a
        for j in range(i + 1, min(i + look_forward + 1, len(df))):
            # Conservative: check the stop first if both are touched in a bar.
            if lows[j] <= sl_price:
                labels[i] = 0
                break
            if highs[j] >= tp_price:
                labels[i] = 1
                break
    return pd.Series(labels, index=df.index)


def train_all_symbols():
    """Train XGBoost and RandomForest experts for all symbols."""
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SUI-USD", "PEPE-USD"]
    
    print("\n🚀 STARTING MEGA-TRAINING (25 Features)")
    print(f"📁 Saving models to: {MODEL_STORAGE_DIR}")
    print(f"📊 Using {len(FEATURE_COLUMNS)} features from config2\n")
    
    for symbol in symbols:
        try:
            path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
            if not os.path.exists(path):
                print(f"  ⚪ Skipped {symbol}: File {path} not found")
                continue

            print(f"🧠 Processing {symbol}...")
            raw_df = pd.read_csv(path)
            df, feats = apply_mega_features(raw_df)
            
            if len(df) < 500:
                print(f"  ⚠️ {symbol} ignored: Only {len(df)} rows left after indicators.")
                continue

            # ATR-based labels aligned to the live bot's 3:1.5 ATR exits (FIX #4),
            # so the experts predict the trade the engine actually takes.
            df['target'] = create_atr_labels(df, look_forward=24, tp_atr=3.0, sl_atr=1.5)

            X = df[feats].values
            y = df['target'].values
            
            # Drop last 24 rows (labels unreliable near end of dataset)
            X = X[:-24]
            y = y[:-24]

            # Time-based train/test split (80/20) — never shuffle time series
            split_idx = int(len(X) * 0.80)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"  📊 Split: {len(X_train)} train / {len(X_test)} test")
            
            train_pos_pct = y_train.mean() * 100
            test_pos_pct = y_test.mean() * 100
            print(f"  ⚖️ Class balance — Train: {train_pos_pct:.1f}% positive | Test: {test_pos_pct:.1f}% positive")

            ticker = symbol.split('-')[0].lower()
            save_path_xgb = os.path.join(MODEL_STORAGE_DIR, f"{ticker}_1h_xgboost_model.joblib")
            save_path_rf = os.path.join(MODEL_STORAGE_DIR, f"{ticker}_1h_randomforest_model.joblib")

            # Train XGBoost (Regularized to reduce overfitting)
            # OLD: max_depth=6, n_estimators=150 → memorized training data
            # NEW: max_depth=3, fewer trees, min_child_weight prevents leaf overfitting
            #      subsample/colsample add randomness to prevent memorization
            #      reg_alpha/reg_lambda add L1/L2 penalties
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=3,              # Was 6 — shallower trees generalize better
                learning_rate=0.05,
                base_score=0.5,
                min_child_weight=10,      # Requires more samples per leaf
                subsample=0.8,            # Only use 80% of rows per tree
                colsample_bytree=0.8,     # Only use 80% of features per tree
                reg_alpha=0.1,            # L1 regularization
                reg_lambda=1.0,           # L2 regularization
                gamma=1.0                 # Minimum loss reduction to split
            )
            xgb.fit(X_train, y_train)
            
            xgb_train_acc = (xgb.predict(X_train) == y_train).mean() * 100
            xgb_test_acc = (xgb.predict(X_test) == y_test).mean() * 100
            print(f"  🌲 XGBoost  — Train: {xgb_train_acc:.1f}% | Test: {xgb_test_acc:.1f}%", end="")
            
            if xgb_train_acc - xgb_test_acc > 10:
                print(f" ⚠️ OVERFIT GAP: {xgb_train_acc - xgb_test_acc:.1f}%")
            else:
                print(" ✅")
            
            joblib.dump({"model": xgb, "feature_names": feats}, save_path_xgb)
            
            # Train RandomForest (Regularized)
            # OLD: max_depth=10 → trees grew until they memorized everything
            # NEW: max_depth=4, min_samples constraints, max_features adds randomness
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,              # Was 10 — much shallower
                min_samples_split=20,     # Need 20+ samples to create a branch
                min_samples_leaf=10,      # Each leaf needs 10+ samples
                max_features='sqrt',      # Only consider sqrt(25)≈5 features per split
            )
            rf.fit(X_train, y_train)
            
            rf_train_acc = (rf.predict(X_train) == y_train).mean() * 100
            rf_test_acc = (rf.predict(X_test) == y_test).mean() * 100
            print(f"  🌳 RF       — Train: {rf_train_acc:.1f}% | Test: {rf_test_acc:.1f}%", end="")
            
            if rf_train_acc - rf_test_acc > 10:
                print(f" ⚠️ OVERFIT GAP: {rf_train_acc - rf_test_acc:.1f}%")
            else:
                print(" ✅")

            # FIX #14: actually persist the RandomForest. Previously only XGBoost
            # was dumped and save_path_rf was computed but never written, so any
            # symbol without a stale RF file ran with 1/3 of the council dead
            # (RandomForest silently returned a constant 0.5).
            joblib.dump({"model": rf, "feature_names": feats}, save_path_rf)

            print(f"  💾 Saved XGB → {save_path_xgb}")
            print(f"  💾 Saved RF  → {save_path_rf}")

        except Exception as e:
            print(f"  ❌ {symbol} CRASH: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    train_all_symbols()
