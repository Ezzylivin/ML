# File: app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from feature_engineering import engineer_features  # ✅ your existing function

print("Initializing Flask app...")
app = Flask(__name__)
CORS(app)

# Load model
try:
    model = joblib.load('models/btc_xgboost_model.joblib')  # adjust path if needed
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found.")
    model = None

@app.route("/")
def root():
    return {"status": "ok", "message": "ML Feature Engineering Service running!"}

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        data = request.get_json()
        candles = data.get("candles")
        if not candles:
            return jsonify({"error": "No candle data provided."}), 400

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_features = engineer_features(df.copy())

        if len(df_features) == 0:
            return jsonify({"signal": 0, "confidence": 0.5, "reason": "Not enough data for features"})

        # Match training columns
        X_live = df_features.drop(columns=["timestamp", "open", "high", "low", "close", "volume"])
        latest_features = X_live.iloc[[-1]]

        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)

        signal = int(prediction[0])
        confidence = float(probabilities[0].max())
        print(f"Prediction successful: Signal={signal}, Confidence={confidence:.2f}")

        return jsonify({"signal": signal, "confidence": confidence})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
