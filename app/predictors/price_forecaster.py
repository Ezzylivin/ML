import os
import logging
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# 🟢 NEW: Foundation & Transformer Libraries
try:
    from tabpfn import TabPFNClassifier
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import lightning.pytorch as pl
except ImportError:
    print("⚠️  Notice: tabpfn or pytorch-forecasting not found. Only XGBoost will be available.")

# Internal Imports from your modular config
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR, ML_CONFIG

logger = logging.getLogger("PriceForecaster")
logging.basicConfig(level=logging.INFO)

class PriceForecaster:
    """
    ENGINE v7.2: Multi-Model Research Engine.
    Features: XGBoost, TabPFN (Foundation), and TFT (Attention).
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
        
        # 2. CREATE TARGET LABEL
        # We predict if price will be HIGHER in 4 bars (Forward-Looking)
        forward_lookup = 4 
        df['future_close'] = df['close'].shift(-forward_lookup)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        df.dropna(inplace=True)
        
        # Features and Target
        X = df.drop(columns=['time', 'future_close', 'target']).select_dtypes(include=[np.number])
        y = df['target']
        
        return X, y, df

    # --- 🟢 MODEL 1: XGBOOST (The Standard) ---
    def train_xgboost(self):
        X, y, _ = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, objective='binary:logistic', random_state=42)
        model.fit(X_train, y_train)
        
        self._save_and_log(model, "XGBoost", y_test, model.predict(X_test))

    # --- 🟢 MODEL 2: TabPFN (The Foundation Model) ---
    def train_tabpfn(self):
        """
        TabPFN doesn't 'train' - it uses In-Context Learning.
        We 'save' a snapshot of the most recent market context to calibrate it.
        """
        X, y, _ = self.prepare_data()
        # TabPFN is optimized for N < 1000
        X_train, X_test, y_train, y_test = train_test_split(X.tail(1000), y.tail(1000), test_size=0.1, shuffle=False)
        
        # Initialize the Prior-Data Fitted Network
        model = TabPFNClassifier(device='cpu') # Use 'cuda' if GPU available
        model.fit(X_train, y_train) 
        
        self._save_and_log(model, "TabPFN", y_test, model.predict(X_test))

    # --- 🟢 MODEL 3: TFT (The Attention Specialist) ---
    def train_tft(self):
        """
        Temporal Fusion Transformer (TFT).
        Captures multi-horizon regimes using self-attention.
        """
        _, _, df = self.prepare_data()
        df["time_idx"] = np.arange(len(df))
        df["group"] = 0 # Single asset training
        
        # Define the TimeSeriesDataSet for PyTorch Forecasting
        max_prediction_length = 4
        max_encoder_length = self.lookback
        
        training_cutoff = df["time_idx"].max() - max_prediction_length
        training_data = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=["close", "rsi_14", "adx_14"], # Add more features here
            allow_missing_timesteps=True
        )
        
        # Train via PyTorch Lightning
        trainer = pl.Trainer(max_epochs=10, accelerator="auto")
        tft = TemporalFusionTransformer.from_dataset(training_data, learning_rate=0.03, hidden_size=16, attention_head_size=4, dropout=0.1, loss=QuantileLoss())
        
        trainer.fit(tft, training_data.to_dataloader(train=True, batch_size=64))
        
        save_path = os.path.join(MODEL_STORAGE_DIR, f"{self.symbol}_{self.timeframe}_Transformer.ckpt")
        tft.save(save_path)
        logger.info(f"💾 Transformer Brain saved to {save_path}")

    def _save_and_log(self, model, name, y_test, preds):
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        model_id = f"{self.symbol}_{self.timeframe}_{name}"
        
        save_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        joblib.dump(model, save_path)
        logger.info(f"✅ {name} Complete. Accuracy: {acc:.2f} | Precision: {prec:.2f} | Path: {save_path}")

if __name__ == "__main__":
    forecaster = PriceForecaster(symbol='BTC-USD', timeframe='1h')
    
    # Choose your fighter
    forecaster.train_xgboost()
    # forecaster.train_tabpfn() # Uncomment to use Foundation Model
    # forecaster.train_tft()    # Uncomment to use Deep Learning Attention
