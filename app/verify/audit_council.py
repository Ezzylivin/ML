import pandas as pd
import numpy as np
from app.predictors.model_factory import ModelFactory

def audit_brains(symbol="BTC-USD"):
    experts = ['XGBoost', 'LSTM', 'RandomForest', 'Transformer', 'Stacking']
    # Create mock data that looks like a technical dataframe
    mock_data = pd.DataFrame(
        np.random.randn(100, 10), 
        columns=['open', 'high', 'low', 'close', 'volume', 'atr_14', 'rsi_14', 'adx_14', 'feature1', 'feature2']
    )
    
    print(f"⚖️ AUDITING COUNCIL BRAINS FOR {symbol}...")
    for e_type in experts:
        try:
            predictor = ModelFactory.load_model(e_type, symbol=symbol)
            brain_class = predictor.__class__.__name__
            
            if brain_class == "DummyPredictor":
                print(f"🔴 {e_type:12}: OFFLINE (Using Dummy 0.5)")
            else:
                prob = predictor.predict_direction(mock_data)
                print(f"🟢 {e_type:12}: ONLINE  ({brain_class}) | Mock Prediction: {prob:.4f}")
        except Exception as e:
            print(f"🟡 {e_type:12}: UNSTABLE ({str(e)[:50]})")

if __name__ == "__main__":
    audit_brains()
