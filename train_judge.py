import os, glob, pandas as pd, numpy as np, joblib, sys
sys.path.append(os.getcwd())

from app.config2 import DATA_DIR
# 🎯 IMPORT: Using the same logic as the experts
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels
from app.predictors.model_factory import ModelFactory
from xgboost import XGBClassifier

def retrain_full_council_judge():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]
    # 🎯 UPGRADE: Added 'transformer' to the council
    expert_types = ["xgboost", "randomforest", "transformer", "lightgbm"]

    for symbol in symbols:
        ticker = symbol.split('-')[0].lower()
        csv_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        
        if not os.path.exists(csv_path): continue
        
        try:
            print(f"\n⚖️ CALIBRATING MASTER JUDGE: {symbol}")
            # 1. Load Data
            df_raw = pd.read_csv(csv_path)
            df, feats = apply_mega_features(df_raw)
            
            # 🎯 FIX: Use the 1% Strategic Labels to match the experts
            df['target'] = create_strategic_labels(df, look_forward=24, tp=1.0, sl=1.0)
            
            # Leave room for the 24-hour target lookahead
            y = df['target'].values[:-24]
            X_for_experts = df.iloc[:-24] 

            # 2. Collect ALL Expert Opinions
            opinions = []
            valid_expert_names = []

            for e_type in expert_types:
                expert = ModelFactory.load_model(e_type, symbol=symbol)
                if expert:
                    print(f"   🎙️ Gathering opinion from: {e_type}")
                    probs = []
                    # Optimization: Only predict on a sample or the recent history if data is huge
                    # but for total parity, we predict on the training window:
                    for i in range(len(X_for_experts)):
                        # RawModelAdapter handles Experts vs Transformers automatically
                        p = expert.predict_direction(df.iloc[:i+1])
                        probs.append(p)
                    
                    opinions.append(probs)
                    valid_expert_names.append(f"{e_type}_score")

            if len(opinions) < 2:
                print(f"   ⚠️ Not enough experts found for {symbol}. Skipping.")
                continue

            # 3. Create the Meta-Feature Matrix
            X_stacked = np.column_stack(opinions)

            # 4. Train the Master Judge (Shallow to prevent overfitting)
            judge = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
            judge.fit(X_stacked, y)

            # 5. Save with Metadata
            save_path = f"app/models/{ticker}_1h_stacking_model.joblib"
            payload = {
                "model": judge, 
                "feature_names": valid_expert_names,
                "is_meta_model": True
            }
            joblib.dump(payload, save_path)
            print(f"   ✅ MASTER JUDGE SAVED: {save_path} (Weighting {len(valid_expert_names)} experts)")

        except Exception as e:
            print(f"   ❌ FAILED {symbol}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    retrain_full_council_judge()
