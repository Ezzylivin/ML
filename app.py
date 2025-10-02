# File: app.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from feature_engineering import engineer_features # ✅ This will now work!

print("Initializing Flask app...")
app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('models/btc_xgboost_model.joblib') # Or your final model name
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    try:
        data = request.get_json()
        candles = data['candles']
        if not candles:
            return jsonify({'error': 'No candle data provided.'}), 400

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # ✅ Call the imported function
        df_features = engineer_features(df.copy())
        
        if len(df_features) == 0:
            return jsonify({'signal': 0, 'confidence': 0.5, 'reason': 'Not enough data for features'})

        # Prepare the final row for prediction
        # Ensure the columns here EXACTLY match the columns used for training
        X_live = df_features.drop(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        latest_features = X_live.iloc[[-1]]
        
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)
        
        signal = int(prediction[0])
        confidence = float(probabilities[0].max())
        
        print(f"Prediction successful: Signal={signal}, Confidence={confidence:.2f}")
        return jsonify({'signal': signal, 'confidence': confidence})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=10000)
