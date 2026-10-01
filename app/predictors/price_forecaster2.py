import os
import logging
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import traceback
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler

# Internal Project Imports
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR, ML_CONFIG
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("PriceForecaster")
logging.basicConfig(level=logging.INFO)

class PriceForecaster:
    """
    ENGINE v7.9: Deep Learning & Ensemble Engine.
    Features: XGBoost, RandomForest, and TensorFlow LSTM.
    """
    def __init__(self, symbol='BTC-USD', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = ML_CONFIG.get('DEFAULT_LOOKBACK_WINDOW', 50)

    def prepare_data(self):
        """Loads CSV and performs dynamic Feature Engineering."""
        path = os.path.join(DATA_DIR, f"{self.symbol}-{self.timeframe}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing training data at {path}")

        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        
        # 1. DYNAMIC TECHNICAL FEATURES
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(append=True)
        df.ta.macd(append=True)
        
        # 2. CREATE TARGET LABEL (4-Bar Forward)
        forward_lookup = 4 
        df['future_close'] = df['close'].shift(-forward_lookup)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        df.dropna(inplace=True)
        
        # 3. ROBUST COLUMN DROPPING (Fixes the KeyError: 'time')
        # We drop any non-numeric columns and our targets
        time_identifiers = ['time', 'timestamp', 'date', 'datetime', 'opened_at']
        cols_to_drop = [c for c in time_identifiers if c in df.columns] + ['future_close', 'target']
        
        X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
        y = df['target']
        
        return X, y, df

    # --- 🟢 MODEL 1: XGBOOST ---
    def train_xgboost(self):
        logger.info("🌲 Training XGBoost Specialist...")
        X, y, _ = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, objective='binary:logistic')
        model.fit(X_train, y_train)
        self._save_and_log(model, "XGBoost", y_test, model.predict(X_test))

    # --- 🔵 MODEL 2: TENSORFLOW LSTM ---
    def train_lstm(self):
        """Sequential Recurrent Neural Network for temporal patterns."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            logger.info("🧠 Training TensorFlow LSTM (Sequential Memory)...")
        except ImportError:
            logger.error("❌ TensorFlow not found. Skipping LSTM.")
            return

        X, y, _ = self.prepare_data()
        
        # LSTM requires scaled data (0 to 1)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create Sliding Window: (Samples, TimeSteps, Features)
        X_3d, y_3d = [], []
        for i in range(self.lookback, len(X_scaled)):
            X_3d.append(X_scaled[i-self.lookback:i])
            y_3d.append(y.iloc[i])
        
        X_3d, y_3d = np.array(X_3d), np.array(y_3d)
        split = int(len(X_3d) * 0.8)

        # Build the Architecture
        model = Sequential([
            Input(shape=(self.lookback, X.shape[1])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') # Probability output
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train
        model.fit(X_3d[:split], y_3d[:split], epochs=10, batch_size=32, verbose=0)
        
        # Save Model and Scaler (Scaler is required for inference!)
        model_path = os.path.join(MODEL_STORAGE_DIR, f"{self.symbol}_{self.timeframe}_LSTM.h5")
        scaler_path = os.path.join(MODEL_STORAGE_DIR, f"{self.symbol}_{self.timeframe}_LSTM_scaler.joblib")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ LSTM Complete. Saved to {model_path}")

    # --- ⚖️ THE META-MODEL: Level 1 "Judge" ---
    def train_stacking_meta_model(self):
        logger.info("⚖️ Initializing Meta-Model Stacking...")
        X, y, _ = self.prepare_data()
        
        # We only use the models we know are working/installed
        expert_types = ['XGBoost', 'LSTM'] 
        meta_features = []
        eval_window = 200
        
        for i in range(len(X) - eval_window, len(X)):
            state = X.iloc[i-self.lookback:i]
            votes = []
            for e_type in expert_types:
                try:
                    expert = ModelFactory.load_model(e_type, self.symbol)
                    votes.append(expert.predict_direction(state))
                except:
                    votes.append(0.5)
            meta_features.append(votes)

        X_meta = np.array(meta_features)
        y_meta = y.tail(eval_window).values

        meta_judge = LogisticRegression(solver='lbfgs')
        meta_judge.fit(X_meta, y_meta)

        save_path = os.path.join(MODEL_STORAGE_DIR, f"MetaModel_{self.symbol}.joblib")
        joblib.dump(meta_judge, save_path)
        logger.info(f"💾 Judge Brain saved to {save_path}")

    def _save_and_log(self, model, name, y_test, preds):
        acc = accuracy_score(y_test, preds)
        model_id = f"{self.symbol}_{self.timeframe}_{name}"
        save_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        joblib.dump(model, save_path)
        logger.info(f"✅ {name} Complete. Acc: {acc:.2f}")

if __name__ == "__main__":
    forecaster = PriceForecaster(symbol='BTC-USD', timeframe='1h')
    
    # 1. Train Experts
    forecaster.train_xgboost()
    forecaster.train_lstm()
    
    # 2. Train the Judge
    forecaster.train_stacking_meta_model()
