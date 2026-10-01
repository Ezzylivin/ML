import pandas as pd
import numpy as np

class DNALogic:
    """
    NEO-V7 Genetic Material Library.
    Upgraded to resolve signals (Directional Bias) instead of raw values.
    """
    @staticmethod
    def get_base_genes():
        return {
            "momentum": ["rsi_signal", "stochk_signal", "macd_cross"],
            "trend": ["adx_trend", "sma_slope_pos", "psar_bullish"],
            "volatility": ["atr_expanding", "bb_expansion", "low_vol_squeeze"],
            "operators": ["AND", "OR", "XOR", "THRESHOLD"]
        }

    @staticmethod
    def resolve_expression(df, gene_name):
        """
        Maps a gene to a Directional Bias (-1 to 1).
        This prevents the 'death spiral' by ensuring indicators are interpreted correctly.
        """
        try:
            # 1. MOMENTUM RESOLVERS (Normalizing OS/OB)
            if gene_name == "rsi_signal":
                # Returns 1 if oversold (<30), -1 if overbought (>70), else 0
                return np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
            
            if gene_name == "macd_cross":
                # Direct Directional Bias from MACD Histogram
                return np.where(df['macd_diff'] > 0, 1, -1)

            # 2. TREND RESOLVERS (Alignment)
            if gene_name == "adx_trend":
                # Only signals if trend is strong (>25)
                return np.where(df['adx'] > 25, 1, 0)
            
            if gene_name == "sma_slope_pos":
                # 1 if price is above SMA 200 (Bullish), -1 if below
                return np.where(df['close'] > df['sma_logic'], 1, -1)

            # 3. VOLATILITY FILTERS (The 'Alpha Shield' Logic)
            if gene_name == "low_vol_squeeze":
                # Signals 1 if volatility is low (potential for breakout)
                return np.where(df['bb_width'] < df['bb_width'].rolling(20).mean(), 1, 0)

            # --- FALLBACK: FUZZY MATCHING FOR RAW COLS ---
            if gene_name in ["AND", "OR", "XOR", "THRESHOLD"]:
                return None
                
            cols = [c for c in df.columns if gene_name.lower() in c.lower()]
            if not cols:
                return pd.Series(0, index=df.index)
                
            # Normalize raw values to a 0-1 scale to prevent extreme outliers
            val = df[cols[0]]
            return (val - val.rolling(50).min()) / (val.rolling(50).max() - val.rolling(50).min() + 1e-9)

        except Exception as e:
            # Silence errors in backtest loop but return neutral
            return pd.Series(0, index=df.index)

    @staticmethod
    def apply_operator(op, signal_a, signal_b):
        """Helper to combine two signals based on the DNA Operator."""
        if op == "AND":
            return np.where((signal_a > 0) & (signal_b > 0), 1, 0)
        if op == "OR":
            return np.where((signal_a > 0) | (signal_b > 0), 1, 0)
        if op == "XOR":
            return np.where((signal_a > 0) ^ (signal_b > 0), 1, 0)
        return signal_a # Default fallback
