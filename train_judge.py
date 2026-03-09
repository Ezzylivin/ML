import os, glob, pandas as pd, numpy as np, joblib, sys
sys.path.append(os.getcwd())
from app.config2 import DATA_DIR
from app.verify.engineer_and_train import apply_mega_features
from app.predictors.model_factory import ModelFactory
from xgboost import XGBClassifier

def retrain_full_council_judge():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]
    # 🎯 The full list of expert types we want the Judge to listen to
    expert_types = ["xgboost", "randomforest", "lightgbm", "mlpclassifier"]

    for symbol in symbols:
        ticker = symbol.split('-')[0].lower()
        csv_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        
        if not os.path.exists(csv_path): continue
        
        try:
            print(f"\n⚖️ CALIBRATING MASTER JUDGE: {symbol}")
            # 1. Load Data
            df_raw = pd.read_csv(csv_path)
            df, feats = apply_mega_features(df_raw)
            
            X_raw = df[feats].values
            y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
            X_for_experts = df.iloc[:-1] # Leave room for target shift

            # 2. Collect ALL Expert Opinions
            opinions = []
            valid_expert_names = []

            for e_type in expert_types:
                # Use our fixed ModelFactory to get the Adapter-wrapped model
                expert = ModelFactory.load_model(e_type, symbol=symbol)
                if expert:
                    print(f"   🎙️ Gathering opinion from: {e_type}")
                    # We generate probabilities for the entire history to train the stacker
                    # Note: In a real prod environment, use cross-validation here to prevent leakage
                    probs = []
                    for i in range(len(X_for_experts)):
                        # Pass the slice up to that point
                        p = expert.predict_direction(df.iloc[:i+1])
                        probs.append(p)
                    
                    opinions.append(probs)
                    valid_expert_names.append(f"{e_type}_score")

            if len(opinions) < 2:
                print(f"   ⚠️ Not enough experts found for {symbol}. Skipping.")
                continue

            # 3. Create the Meta-Feature Matrix
            # Columns: [XGB_prob, RF_prob, LGBM_prob, MLP_prob]
            X_stacked = np.column_stack(opinions)

            # 4. Train the Master Judge
            # We use a shallow XGBoost to prevent the Judge from overfitting to one expert
            judge = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
            judge.fit(X_stacked, y)

            # 5. Save with Metadata
            save_path = f"app/models/{ticker}_1h_stacking_model.joblib"
            payload = {
                "model": judge, 
                "feature_names": valid_expert_names, # ['xgboost_score', 'randomforest_score', ...]
                "is_meta_model": True
            }
            joblib.dump(payload, save_path)
            print(f"   ✅ MASTER JUDGE SAVED: {save_path} (listening to {len(valid_expert_names)} experts)")

        except Exception as e:
            print(f"   ❌ FAILED {symbol}: {e}")

if __name__ == "__main__":
    retrain_full_council_judge()
