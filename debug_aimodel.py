import pandas as pd
import joblib
import pandas_ta as ta
import numpy as np

# 🟢 CONFIG
MODEL_PATH = "app/models/sol_1h_stacking_model.joblib"

print(f"🔍 Inspecting: {MODEL_PATH}")

try:
    # 1. Load Model
    payload = joblib.load(MODEL_PATH)
    print("✅ Model Loaded!")
    print(f"   Type: {type(payload)}")
    
    if isinstance(payload, dict):
        required_features = payload.get('feature_names', [])
        print(f"   Required Features ({len(required_features)}): {required_features}")
    else:
        print("   ⚠️ No metadata found (Legacy format).")
        required_features = []

    # 2. Create Dummy Data (Exact structure as Backtester)
    # We create 100 rows to allow indicators to calculate
    df = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    })
    
    # 3. Calculate Indicators (Matching Backtester)
    print("⚗️ Calculating Indicators...")
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

    # 4. Prepare Single Row for Prediction
    last_row = df.iloc[[-1]].copy()
    print(f"   Available Columns in Data: {list(last_row.columns)}")

    # 5. Check for Mismatch
    if required_features:
        missing = [f for f in required_features if f not in last_row.columns]
        if missing:
            print(f"❌ FATAL ERROR: Data is missing these columns needed by model:\n   {missing}")
        else:
            print("✅ All columns match!")
            
            # 6. Try Prediction
            model = payload['model']
            scaler = payload.get('scaler')
            X = last_row[required_features]
            
            if scaler:
                print("   ⚖️ Scaling data...")
                X = scaler.transform(X)
                
            print("   🔮 Attempting Prediction...")
            prob = model.predict_proba(X)
            print(f"   ✅ SUCCESS! Prediction Output: {prob}")

except Exception as e:
    print(f"\n🚨 CRASHED: {e}")
