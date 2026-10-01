import os, glob, pandas as pd, numpy as np, joblib, sys
sys.path.append(os.getcwd())

from app.config2 import DATA_DIR
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels
from app.predictors.model_factory import ModelFactory
from xgboost import XGBClassifier

# 🎯 CONFIG: 3 Years of hourly data
LOOKBACK_WINDOW = 26280 # 24 * 365 * 3
FEATS = ['open','high','low','close','volume','sma_50','sma_200','ema_9','ema_21','ema_20',
         'rsi','atr','adx','st_trend','BBL_20_2.0_2.0','BBU_20_2.0_2.0','STOCHk_14_3_3',
         'MACD_12_26_9','MACDs_12_26_9','pa_high','pa_low','vol_ma','adx_logic','atr_logic','sma_logic']

def retrain_turbo_judge():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]
    expert_types = ["xgboost", "randomforest", "transformer"]

    for symbol in symbols:
        ticker = symbol.split('-')[0].lower()
        csv_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        if not os.path.exists(csv_path): continue
        
        try:
            print(f"\n⚖️  TURBO CALIBRATION: {symbol} (Last 3 Years)")
            df_raw = pd.read_csv(csv_path)
            df, _ = apply_mega_features(df_raw)
            
            # 1. Slice for the last 3 years + padding for targets
            df = df.tail(LOOKBACK_WINDOW + 24)
            df['target'] = create_strategic_labels(df, look_forward=24, tp=1.0, sl=1.0)
            
            y = df['target'].values[:-24]
            X_df = df.iloc[:-24]

            # 2. Collect Opinions via Batch Prediction
            opinions = []
            valid_experts = []

            for e_type in expert_types:
                expert = ModelFactory.load_model(e_type, symbol)
                if not expert: continue
                
                print(f"   🎙️  Batch Polling: {e_type}...")
                
                if e_type == "transformer":
                    # Transformers require sliding window sequences
                    # We build a 3D matrix (N, 50, 25) and predict in one go
                    X_raw = X_df[FEATS].values
                    X_seq = []
                    for i in range(50, len(X_raw)):
                        X_seq.append(X_raw[i-50:i])
                    X_seq = np.array(X_seq)
                    
                    # Predict all sequences at once (Much faster than the loop)
                    # We pad the beginning with 0.5 for the first 50 rows
                    nn_probs = expert.model.predict(X_seq, batch_size=256, verbose=0).flatten()
                    probs = np.concatenate([np.full(50, 0.5), nn_probs])
                else:
                    # XGB/RF are lightning fast with 2D matrices
                    probs = expert.model.predict_proba(X_df[FEATS])[:, 1]
                
                opinions.append(probs)
                valid_experts.append(f"{e_type}_score")

            # 3. Training the Meta-Model
            X_meta = np.column_stack(opinions)
            judge = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
            judge.fit(X_meta, y)

            # 4. Save
            save_path = f"app/models/{ticker}_1h_stacking_model.joblib"
            joblib.dump({"model": judge, "feature_names": valid_experts, "is_meta_model": True}, save_path)
            print(f"   ✅ MASTER JUDGE SAVED: {save_path}")

        except Exception as e:
            print(f"   ❌ FAILED {symbol}: {e}")

if __name__ == "__main__":
    retrain_turbo_judge()
