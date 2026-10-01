import joblib
import os

# Check the BTC Judge
meta_path = "/root/Project/ML/data/saved_models/MetaModel_BTC-USD.joblib"

if os.path.exists(meta_path):
    judge = joblib.load(meta_path)
    # n_features_in_ tells us how many experts this Judge expects
    expected = judge.n_features_in_
    print(f"⚖️  JUDGE AUDIT: {meta_path}")
    print(f"📊 This Judge expects exactly {expected} experts.")
    
    if expected == 3:
        print("❌ ERROR: Your live system is sending 4 (XGB, LSTM, RF, TRANS).")
        print("💡 FIX: We must disable the 'LSTM' alias so the Judge only sees 3.")
else:
    print("❌ MetaModel file not found.")
