# File: feature_engineering.py

import pandas_ta as ta

def engineer_features(df):
    """
    Takes a raw OHLCV DataFrame and adds all the necessary feature columns for the ML model.
    This must be the *exact same* feature set the model was trained on.
    """
    print("Engineering features for live data...")
    
    # Add all your pandas-ta indicators here
    df.ta.rsi(period=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.cci(period=20, append=True)
    df.ta.bbands(period=20, std=2, append=True)
    df.ta.atr(period=14, append=True)
    df.ta.sma(period=50, append=True)
    df.ta.sma(period=200, append=True)
    df.ta.ichimoku(append=True)
    df.ta.obv(append=True)
    
    # Add any custom features
    df['sma_crossover_signal'] = (df['SMA_50'] > df['SMA_200']).astype(int).replace(0, -1)
    
    # IMPORTANT: Drop rows with NaN values created by the indicators
    df.dropna(inplace=True)
    
    print("Features engineered successfully.")
    return df
