import pandas as pd
import numpy as np
import pandas_ta as ta

LOOK_FORWARD_CANDLES = 24
PROFIT_TARGET_PCT = 1.0
STOP_LOSS_PCT = 1.0

def create_target_labels(df, look_forward, profit_target_pct, stop_loss_pct):
    target = np.zeros(len(df))
    profit_target = 1 + (profit_target_pct / 100)
    stop_loss_target = 1 - (stop_loss_pct / 100)
    
    for i in range(len(df) - look_forward):
        entry_price = df['close'].iloc[i]
        future_window = df.iloc[i + 1: i + 1 + look_forward]

        profit_hit_mask = future_window['high'] >= entry_price * profit_target
        stop_loss_hit_mask = future_window['low'] <= entry_price * stop_loss_target

        first_profit_hit = np.where(profit_hit_mask)[0]
        first_stop_loss_hit = np.where(stop_loss_hit_mask)[0]

        if len(first_profit_hit) > 0 and len(first_stop_loss_hit) > 0:
            target[i] = 1 if first_profit_hit[0] < first_stop_loss_hit[0] else -1
        elif len(first_profit_hit) > 0:
            target[i] = 1
        elif len(first_stop_loss_hit) > 0:
            target[i] = -1
    return target

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators, signals, and targets to raw OHLCV data."""

    # === Technical Indicators ===
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.cci(length=20, append=True)

    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.psar(append=True)
    df.ta.ichimoku(append=True)

    df.ta.obv(append=True)

    # Replace NaNs with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # === Helper to find columns dynamically ===
    def find_col(df, keyword):
        cols = [c for c in df.columns if keyword.upper() in c.upper()]
        if not cols:
            raise ValueError(f"Column containing '{keyword}' not found!")
        return cols[0]

    # === Signals ===
    df['atr_signal'] = np.where(df['close'] > df[find_col(df, 'ATR')], 1, -1)
    df['bb_signal'] = np.where(df['close'] > df[find_col(df, 'BBU')], -1, 
                        np.where(df['close'] < df[find_col(df, 'BBL')], 1, 0))
    df['cci_signal'] = np.where(df[find_col(df, 'CCI')] > 100, 1, 
                         np.where(df[find_col(df, 'CCI')] < -100, -1, 0))
    df['ichimoku_signal'] = np.where(df['close'] > df[find_col(df, 'ISA')], 1, -1)
    df['macd_signal'] = np.where(df[find_col(df, 'MACDh')] > 0, 1, -1)
    df['obv_signal'] = np.where(df[find_col(df, 'OBV')] > df[find_col(df, 'OBV')].shift(1), 1, -1)
    df['psar_signal'] = np.where(df[find_col(df, 'PSARl')] < df['close'], 1, -1)
    df['rsi_signal'] = np.where(df[find_col(df, 'RSI')] > 70, -1, 
                         np.where(df[find_col(df, 'RSI')] < 30, 1, 0))
    df['sma_crossover_signal'] = np.where(df[find_col(df, 'SMA_50')] > df[find_col(df, 'SMA_200')], 1, -1)
    df['stoch_signal'] = np.where(df[find_col(df, 'STOCHk')] > 80, -1,
                           np.where(df[find_col(df, 'STOCHk')] < 20, 1, 0))

    # Combined momentum signal
    signal_columns = [
        'atr_signal','bb_signal','cci_signal','ichimoku_signal',
        'macd_signal','obv_signal','psar_signal','rsi_signal',
        'sma_crossover_signal','stoch_signal'
    ]
    df['momentum_strength'] = df[signal_columns].sum(axis=1)

    # === Target Labels ===
    df['target'] = create_target_labels(df, LOOK_FORWARD_CANDLES, PROFIT_TARGET_PCT, STOP_LOSS_PCT)

    # Drop warmup rows
    df = df.dropna().reset_index()

    return df
