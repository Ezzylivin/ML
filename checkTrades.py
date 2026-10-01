import pandas as pd
import pandas_ta as ta
import joblib
import os
import sys

# 🟢 CONFIG
MODEL_PATH = "app/models/sol_1h_stacking_model.joblib"
DATA_PATH = "data/SOL-USD-1h.csv"

print(f"🔍 DIAGNOSTIC MODE: Checking {MODEL_PATH}...")

# 1. Load Data
if not os.path.exists(DATA_PATH):
    print("❌ Data file missing!")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# 2. Calculate Indicators (Exact match to Backtester)
df['sma_50'] = ta.sma(df['close'], length=50)
df['sma_200'] = ta.sma(df['close'], length=200)
st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
df['st_trend'] = st.iloc[:, 1] if st is not None else 0
df['rsi'] = ta.rsi(df['close'], length=14)
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
adx = ta.adx(df['high'], df['low'], df['close'], length=14)
df['adx'] = adx.iloc[:, 0] if adx is not None else 0
bb = ta.bbands(df['close'], length=20, std=2)
df = pd.concat([df, bb], axis=1)

# 3. Load Model
if not os.path.exists(MODEL_PATH):
    print("❌ Model file missing!")
    sys.exit(1)

payload = joblib.load(MODEL_PATH)
print("✅ Model Loaded.")

if not isinstance(payload, dict):
    print("❌ FATAL: Model is LEGACY format (No metadata).")
    print("   👉 Run: python3 app/services/create_judges.py")
    sys.exit(1)

required_features = payload.get('feature_names', [])
print(f"📋 Model wants these {len(required_features)} columns:")
print(f"   {required_features}")

# 4. Check for Mismatch
last_row = df.iloc[[-1]]
available_cols = list(last_row.columns)

missing = [f for f in required_features if f not in available_cols]

if missing:
    print("\n❌ CRITICAL ERROR: Column Name Mismatch!")
    print(f"   Missing: {missing}")
    print(f"   Available BBL/BBU columns: {[c for c in available_cols if 'BB' in c]}")
    print("\n💡 SOLUTION: The model expects different names than pandas_ta is generating.")
else:
    print("\n✅ COLUMNS MATCH!")
    
    # 5. Test Prediction
    try:
        model = payload['model']
        X = last_row[required_features]
        prob = model.predict_proba(X)[0]
        print(f"🔮 Test Prediction Probability: {prob}")
        if len(prob) == 3:
            print(f"   Buy Probability: {prob[2]:.4f}")
        else:
            print(f"   Buy Probability: {prob[1]:.4f}")
            
        print("\n🚀 CONCLUSION: The system is HEALTHY. The backtester should work.")
    except Exception as e:
        print(f"❌ Prediction Crashed: {e}")
