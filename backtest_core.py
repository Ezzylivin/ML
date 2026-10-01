"""
Shared backtesting and feature-engineering logic, safely isolated from FastAPI app to prevent circular imports.
"""

# This is the proof that the new file is loading
print("--- LOADING FINAL BACKTEST_CORE.PY (v5) ---")

import pandas as pd
import numpy as np
import typing
import pandas_ta as ta
import logging
from config.constants import SLIPPAGE_PCT, RISK_FREE_RATE, MAX_LEVERAGE, EPSILON

def find_col(df, key, exclude=None):
    key = key.upper()
    exclude = exclude.upper() if exclude else ''
    for col in df.columns:
        col_upper = col.upper()
        if key in col_upper and not (exclude and exclude in col_upper):
            return col
    raise KeyError(
        f"API: Could not find required col for key: {key} in columns: {df.columns.tolist()}")


def engineer_features_for_backtest(df: pd.DataFrame, trend_filter_period: typing.Optional[int]) -> pd.DataFrame:
    # (Your full 100+ line function)
    df = df.copy()
    if df['close'].isnull().any():
        df['close'] = df['close'].fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 

    # === 1. Base Technical Indicators ===
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True, col_names=('ATR_14'))
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.psar(append=True)
    df.ta.ichimoku(conversion=9, base=26, span=52, append=True)
    df.ta.obv(append=True)
    df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()
    df.ta.adx(length=14, append=True)
    
    if trend_filter_period and trend_filter_period > 0:
        sma_trend_col = f'SMA_{trend_filter_period}'
        df.ta.sma(length=trend_filter_period,
                     append=True, col_names=(sma_trend_col))

    # === 2. Get Column Names ===
    try:
        rsi_col = find_col(df, 'RSI_14')
        macd_col = find_col(df, 'MACD_12_26_9', 'MACDS')
        macd_signal_col = find_col(df, 'MACDS_12_26_9')
        stoch_k_col = find_col(df, 'STOCHK_14_3_3')
        stoch_d_col = find_col(df, 'STOCHD_14_3_3')
        cci_col = find_col(df, 'CCI_20_0.015')
        bbl_col = find_col(df, 'BBL_20_2.0')
        bbu_col = find_col(df, 'BBU_20_2.0')
        bbm_col = find_col(df, 'BBM_20_2.0')
        atr_col = find_col(df, 'ATR_14')
        sma10_col = find_col(df, 'SMA_10')
        sma50_col = find_col(df, 'SMA_50')
        sma200_col = find_col(df, 'SMA_200')
        psar_col = find_col(df, 'PSARr')
        adx_col = find_col(df, 'ADX_14')
        obv_col = find_col(df, 'OBV')
    except KeyError as e:
        logging.error(f"Failed to find a base TA column: {e}. Stopping.")
        raise e

    # === 3. New Relational & Contextual Features ===
    df['price_vs_sma50'] = (df['close'] - df[sma50_col]) / (df[sma50_col] + EPSILON)
    df['price_vs_sma200'] = (df['close'] - df[sma200_col]) / (df[sma200_col] + EPSILON)
    df['sma10_vs_sma50'] = (df[sma10_col] - df[sma50_col]) / (df[sma50_col] + EPSILON)
    df['sma50_vs_sma200'] = (df[sma50_col] - df[sma200_col]) / (df[sma200_col] + EPSILON)
    df['price_vs_bbu'] = (df[bbu_col] - df['close']) / (df['close'] + EPSILON)
    df['price_vs_bbl'] = (df[bbl_col] - df['close']) / (df['close'] + EPSILON)
    df['bb_width'] = (df[bbu_col] - df[bbl_col]) / (df[bbm_col] + EPSILON)
    df['rsi_state'] = np.select(
        [df[rsi_col] > 70, df[rsi_col] < 30], [1, -1], 0)
    df['macd_hist_norm'] = (df[macd_col] - df[macd_signal_col]) / (df['close'] + EPSILON)
    df['stoch_cross'] = np.sign(df[stoch_k_col] - df[stoch_d_col]).diff()
    df['cci_state'] = np.select(
        [df[cci_col] > 100, df[cci_col] < -100], [1, -1], 0)
    df['atr_pct'] = (df[atr_col] / (df['close'] + EPSILON)) * 100
    df['adx_strong_trend'] = (df[adx_col] > 25).astype(int)
    
    df['rsi_roc_3'] = (df[rsi_col] - df[rsi_col].shift(3)) / (df[rsi_col].shift(3) + EPSILON)
    df['volume_roc_10'] = (df['volume'] - df['volume'].shift(10)) / (df['volume'].shift(10) + EPSILON)
    df['price_roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + EPSILON)

    # === 4. Create NON-Leaky Lag Features ===
    lag_features_to_create = [
        'RSI_14', 'ADX_14', 'OBV', 'rsi_state', 'atr_pct', 'price_vs_sma200', 
        'macd_hist_norm', 'volume_roc_10', 'price_roc_5', 'bb_width', 
        'adx_strong_trend', 'sma50_vs_sma200', 'stoch_cross', 'cci_state', 'rsi_roc_3'
    ]
    
    final_lag_list = []
    for feat_base in lag_features_to_create:
        try:
            col_name = find_col(df, feat_base)
            if col_name not in final_lag_list:
                final_lag_list.append(col_name)
        except KeyError:
            logging.warning(f"Could not find base feature '{feat_base}' for lagging.")
            
    for feat_col in final_lag_list:
        if feat_col in df.columns:
            for lag in [1, 2, 3]: # Create _lag1, _lag2, and _lag3
                df[f"{feat_col}_lag{lag}"] = df[feat_col].shift(lag)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df

def generate_ta_signals(df: pd.DataFrame, strategy_code: str) -> pd.DataFrame:
    # (Your full 100+ line function)
    df = df.copy()
    df['ta_signal'] = 0
    try:
        rsi_col = find_col(df, 'RSI_14')
        macd_col = find_col(df, 'MACD_12_26_9', 'MACDS')
        macd_signal_col = find_col(df, 'MACDS_12_26_9')
        stoch_k_col = find_col(df, 'STOCHK_14_3_3')
        stoch_d_col = find_col(df, 'STOCHD_14_3_3')
        cci_col = find_col(df, 'CCI_20_0.015')
        bbl_col = find_col(df, 'BBL_20_2.0')
        bbu_col = find_col(df, 'BBU_20_2.0')
        atr_col = find_col(df, 'ATR_14')
        sma10_col = find_col(df, 'SMA_10')
        sma50_col = find_col(df, 'SMA_50')
        psar_col = find_col(df, 'PSARr')
        tenkan_col = find_col(df, 'ITS_9')
        kijun_col = find_col(df, 'IKS_26')
        spanA_col = find_col(df, 'ISA_9')
        spanB_col = find_col(df, 'ISB_26')
        obv_col = find_col(df, 'OBV')
        
        df['close_prev'] = df['close'].shift(1)
        df['rsi_prev'] = df[rsi_col].shift(1)
        df['cci_prev'] = df[cci_col].shift(1)
        df['sma10_prev'] = df[sma10_col].shift(1)
        df['sma50_prev'] = df[sma50_col].shift(1)
        df['macd_hist'] = df[macd_col] - df[macd_signal_col]
        df['macd_hist_prev'] = df['macd_hist'].shift(1)
        df['stoch_k_prev'] = df[stoch_k_col].shift(1)
        df['stoch_d_prev'] = df[stoch_d_col].shift(1)
        df['atr_prev'] = df[atr_col].shift(1)
        df['obv_prev'] = df[obv_col].shift(1)

        if strategy_code == "sma_crossover":
            df.loc[(df['sma10_prev'] <= df['sma50_prev']) & (df[sma10_col] > df[sma50_col]), 'ta_signal'] = 1
            df.loc[(df['sma10_prev'] >= df['sma50_prev']) & (df[sma10_col] < df[sma50_col]), 'ta_signal'] = -1 
        elif strategy_code == "rsi_divergence":
            df.loc[(df['rsi_prev'] >= 30) & (df[rsi_col] < 30), 'ta_signal'] = 1
            df.loc[(df['rsi_prev'] <= 70) & (df[rsi_col] > 70), 'ta_signal'] = -1
        elif strategy_code == "macd_crossover":
            df.loc[(df['macd_hist_prev'] <= 0) & (df['macd_hist'] > 0), 'ta_signal'] = 1
            df.loc[(df['macd_hist_prev'] >= 0) & (df['macd_hist'] < 0), 'ta_signal'] = -1
        elif strategy_code == "stochastic_crossover":
            df.loc[(df['stoch_k_prev'] <= df['stoch_d_prev']) & (df[stoch_k_col] > df[stoch_d_col]), 'ta_signal'] = 1
            df.loc[(df['stoch_k_prev'] >= df['stoch_d_prev']) & (df[stoch_k_col] < df[stoch_d_col]), 'ta_signal'] = -1
        elif strategy_code == "cci_oversold":
            df.loc[(df['cci_prev'] >= -100) & (df[cci_col] < -100), 'ta_signal'] = 1
            df.loc[(df['cci_prev'] <= 100) & (df[cci_col] > 100), 'ta_signal'] = -1
        elif strategy_code == "bollinger_bands":
            df.loc[(df['close_prev'] >= df[bbl_col].shift(1)) & (df['close'] < df[bbl_col]), 'ta_signal'] = 1
            df.loc[(df['close_prev'] <= df[bbu_col].shift(1)) & (df['close'] > df[bbu_col]), 'ta_signal'] = -1 
        elif strategy_code == "ichimoku_cloud":
            df.loc[((df['close_prev'] <= df[spanA_col].shift(1)) | (df['close_prev'] <= df[spanB_col].shift(1))) & ((df['close'] > df[spanA_col]) & (df['close'] > df[spanB_col])), 'ta_signal'] = 1
            df.loc[((df['close_prev'] >= df[spanA_col].shift(1)) | (df['close_prev'] >= df[spanB_col].shift(1))) & ((df['close'] < df[spanA_col]) & (df['close'] < df[spanB_col])), 'ta_signal'] = -1
        elif strategy_code == "atr_signal":
            df.loc[(df['atr_prev'] >= df[atr_col]), 'ta_signal'] = -1
            df.loc[(df['atr_prev'] < df[atr_col]), 'ta_signal'] = 1
        elif strategy_code == "obv_signal":
            df.loc[(df['obv_prev'] >= df[obv_col]), 'ta_signal'] = -1
            df.loc[(df['obv_prev'] < df[obv_col]), 'ta_signal'] = 1
        elif strategy_code == "psar_signal":
            df['ta_signal'] = df[psar_col].fillna(0)
        else:
            logging.warning(f"Strategy code '{strategy_code}' not found. Defaulting to 'Hold' (0).")
    except KeyError as e:
        logging.error(f"KeyError in generate_ta_signals: {e}. Defaulting to 'Hold' (0).")
    except Exception as e:
        logging.error(f"Error in generate_ta_signals: {e}. Defaulting to 'Hold' (0).")
    return df


def run_backtest(
    df: pd.DataFrame,
    signal_column: str,
    initial_balance: float,
    fee: float,
    stop_loss_pct: typing.Optional[float],
    take_profit_pct: typing.Optional[float],
    risk_mode: str,
    risk_percent: float,
    growth_target: typing.Optional[float],
    min_atr_pct: typing.Optional[float],
    trend_filter_period: typing.Optional[int],
    ml_mode: str,
    min_adx_level: typing.Optional[float],
    trailing_stop_atr_mult: typing.Optional[float],
    trailing_stop_pct: typing.Optional[float],
    max_leverage: float = 2.0,
    **kwargs
) -> typing.Dict[str, typing.Any]:
    
    # (Your full 300+ line function)
    # --- 1. Prep ---
    if signal_column not in df.columns:
        raise ValueError(f"Signal column '{signal_column}' not found.")
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        raise ValueError("DF missing OHLC.")
    
    try:
        atr_col_sim = find_col(df, 'ATR_14')
    except KeyError:
        atr_col_sim = None
        if (min_atr_pct is not None and min_atr_pct > 0) or \
            (trailing_stop_atr_mult is not None and trailing_stop_atr_mult > 0):
                raise ValueError("ATR_14 column required for Volatility Filter or ATR Trailing Stop.")
                
    try:
        adx_col_sim = find_col(df, 'ADX_14')
    except KeyError:
        adx_col_sim = None
        if min_adx_level is not None and min_adx_level > 0:
            raise ValueError("ADX_14 column required for ADX Chop Filter.")

    try:
        sma_trend_col = find_col(
            df, f'SMA_{trend_filter_period}') if trend_filter_period else None
    except KeyError:
        sma_trend_col = None
    if trend_filter_period and sma_trend_col and sma_trend_col not in df.columns:
        logging.warning(
            f"Trend Filter column SMA_{trend_filter_period} not found. Filter disabled.")

    # --- 2. Simulation Setup ---
    balance = initial_balance
    equity = initial_balance
    position = 0
    position_size = 0.0
    entry_price = 0.0
    entry_time = None
    equity_curve = []
    trades = []
    stop_loss_price = 0.0
    take_profit_price = 0.0
    current_risk_percent = risk_percent
    
    df_signals = df[signal_column].to_numpy()
    df_low = df["low"].to_numpy()
    df_high = df["high"].to_numpy()
    df_close = df["close"].to_numpy()
    df_open = df["open"].to_numpy()
    df_atr = df[atr_col_sim].to_numpy() if atr_col_sim else np.zeros(len(df))
    df_adx = df[adx_col_sim].to_numpy() if adx_col_sim else np.zeros(len(df))
    df_trend_sma = df[sma_trend_col].to_numpy(
    ) if sma_trend_col and sma_trend_col in df.columns else np.zeros(len(df))

    # --- 3. Main Trading Loop ---
    for i in range(1, len(df) - 1):  # Loop from 1 to second-to-last

        current_low = df_low[i]
        current_high = df_high[i]
        current_close = df_close[i]
        current_atr = df_atr[i]
        current_adx = df_adx[i]
        next_open = df_open[i+1]
        current_timestamp = df.index[i]

        # --- A. Check for Exits (if in a position) ---
        if position != 0:
            exit_price = 0.0
            pnl_reason = "Signal"
            
            if position == 1:
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                    new_stop_loss = current_close - (current_atr * trailing_stop_atr_mult)
                    stop_loss_price = max(stop_loss_price, new_stop_loss)
                elif trailing_stop_pct and trailing_stop_pct > 0: 
                    new_stop_loss = current_close * (1 - trailing_stop_pct / 100.0)
                    stop_loss_price = max(stop_loss_price, new_stop_loss)
            elif position == -1:
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                    new_stop_loss = current_close + (current_atr * trailing_stop_atr_mult)
                    stop_loss_price = min(stop_loss_price, new_stop_loss)
                elif trailing_stop_pct and trailing_stop_pct > 0:
                    new_stop_loss = current_close * (1 + trailing_stop_pct / 100.0)
                    stop_loss_price = min(stop_loss_price, new_stop_loss)

            if position == 1:
                sl_hit = current_low <= stop_loss_price
                tp_hit = take_profit_pct is not None and current_high >= take_profit_price
                signal_exit = (df_signals[i] == -1)

                if sl_hit and tp_hit:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss (Pessimistic)"
                elif sl_hit:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = take_profit_price
                    pnl_reason = "Take Profit"
                elif signal_exit:
                    exit_price = next_open * (1 - SLIPPAGE_PCT)
                    pnl_reason = "Signal"

            elif position == -1:
                sl_hit = current_high >= stop_loss_price
                tp_hit = take_profit_pct is not None and current_low <= take_profit_price
                signal_exit = (df_signals[i] == 1)

                if sl_hit and tp_hit:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss (Pessimistic)"
                elif sl_hit:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = take_profit_price
                    pnl_reason = "Take Profit"
                elif signal_exit:
                    exit_price = next_open * (1 + SLIPPAGE_PCT)
                    pnl_reason = "Signal"

            if exit_price > 0:
                if position == 1:
                    sell_value = position_size * exit_price
                    buy_value = position_size * entry_price
                    net_sell_value = sell_value * (1 - fee)
                    profit_usd = net_sell_value - buy_value
                    balance += net_sell_value
                elif position == -1:
                    buy_value = position_size * exit_price
                    sell_value = position_size * entry_price
                    net_buy_value = buy_value * (1 + fee)
                    profit_usd = sell_value - net_buy_value
                    balance -= net_buy_value

                pnl_pct = (profit_usd / ((position_size * entry_price) + EPSILON)) * 100
                
                if pnl_pct > 1000.0: pnl_pct = 1000.0
                if pnl_pct < -100.0: pnl_pct = -100.0

                trades.append({
                    "action": "sell" if position == 1 else "cover",
                    "price": exit_price,
                    "time": current_timestamp,
                    "size": position_size,
                    "pnl_pct": pnl_pct,
                    "profit_usd": profit_usd,
                    "reason": pnl_reason,
                    "entryPrice": entry_price, 
                    "entryTime": entry_time,   
                    "position": "long" if position == 1 else "short",
                    "result": "win" if profit_usd > 0 else "loss" if profit_usd < 0 else "breakeven"
                })
                position = 0
                position_size = 0.0
                entry_price = 0.0
                entry_time = None 
                stop_loss_price = 0.0
                take_profit_price = 0.0
                if risk_mode == 'dynamic' and growth_target is not None and balance >= growth_target:
                    current_risk_percent = risk_percent

        # --- B. Check for Entries (if flat) ---
        if position == 0 and balance > 0:
            signal = df_signals[i]

            passes_vol_filter = True
            if min_atr_pct is not None and min_atr_pct > 0:
                atr_percentage = (current_atr / (current_close + EPSILON)) * 100 if current_close > 0 else 0
                if atr_percentage < min_atr_pct:
                    passes_vol_filter = False

            passes_trend_filter = True
            if sma_trend_col and sma_trend_col in df.columns:
                sma_value = df_trend_sma[i]
                if (signal == 1 and current_close < sma_value) or \
                   (signal == -1 and current_close > sma_value):
                    passes_trend_filter = False

            passes_adx_filter = True
            if min_adx_level is not None and min_adx_level > 0:
                if current_adx < min_adx_level:
                    passes_adx_filter = False

            if (signal == 1 or signal == -1) and passes_vol_filter and passes_trend_filter and passes_adx_filter:
                trade_price = next_open
                if trade_price == 0:
                    continue

                risk_amount_usd = 0.0
                KELLY_GROWTH_FRACTION = 0.1 

                if risk_mode == 'dynamic' and growth_target is not None and balance < growth_target:
                    distance_to_target = growth_target - balance
                    risk_amount_usd = min(distance_to_target * KELLY_GROWTH_FRACTION, balance * 0.99)
                else:
                    risk_amount_usd = balance * (current_risk_percent / 100.0)

                sl_distance_usd = 0
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                    sl_distance_usd = current_atr * trailing_stop_atr_mult
                elif trailing_stop_pct and trailing_stop_pct > 0:
                    sl_distance_usd = trade_price * (trailing_stop_pct / 100.0)
                elif stop_loss_pct and stop_loss_pct > 0:
                        sl_distance_usd = trade_price * (stop_loss_pct / 100.0)
                
                if sl_distance_usd > EPSILON:
                    calculated_size = risk_amount_usd / sl_distance_usd
                else:
                    calculated_size = risk_amount_usd / (trade_price + EPSILON)

                max_size_by_leverage = (balance * MAX_LEVERAGE) / (trade_price + EPSILON)
                position_size = min(calculated_size, max_size_by_leverage)

                if position_size * trade_price > balance and signal == 1:
                    position_size = balance / (trade_price + EPSILON)

                if position_size <= 1e-9:
                    continue

                entry_time = df.index[i+1]
                if signal == 1:
                    position = 1
                    entry_price = trade_price * (1 + SLIPPAGE_PCT)
                    balance -= position_size * entry_price
                    
                    if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                        stop_loss_price = entry_price - (current_atr * trailing_stop_atr_mult)  
                    elif trailing_stop_pct and trailing_stop_pct > 0:
                        stop_loss_price = entry_price * (1 - trailing_stop_pct / 100.0)
                    elif stop_loss_pct:
                        stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
                    if take_profit_pct:
                        take_profit_price = entry_price * (1 + take_profit_pct / 100.0)
                    
                    trades.append({"action": "buy", "price": entry_price, "time": entry_time, "size": position_size})

                elif signal == -1:
                    position = -1
                    entry_price = trade_price * (1 - SLIPPAGE_PCT)
                    balance += position_size * entry_price

                    if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                        stop_loss_price = entry_price + (current_atr * trailing_stop_atr_mult)  
                    elif trailing_stop_pct and trailing_stop_pct > 0:
                        stop_loss_price = entry_price * (1 + trailing_stop_pct / 100.0)
                    elif stop_loss_pct:
                        stop_loss_price = entry_price * (1 + stop_loss_pct / 100.0)
                    if take_profit_pct:
                        take_profit_price = entry_price * (1 - take_profit_pct / 100.0)
                    
                    trades.append({"action": "sell_short", "price": entry_price, "time": entry_time, "size": position_size})

        # --- C. Update Equity Curve (End of bar) ---
        equity = balance
        if position == 1:
            equity += (position_size * current_close)
        elif position == -1:
            liability = position_size * current_close
            equity = balance - liability
            
        equity_curve.append({"timestamp": current_timestamp, "balance": equity})

    # --- 4. Calculate Final Metrics ---
    final_balance = equity_curve[-1]['balance'] if equity_curve else initial_balance
    total_return = (final_balance / initial_balance - 1) * 100

    equity_series = pd.Series([e['balance'] for e in equity_curve])
    if equity_series.empty:
        equity_series = pd.Series([initial_balance])
        
    daily_returns = equity_series.pct_change().fillna(0)

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / (peak + EPSILON)
    max_drawdown = abs(drawdown.min() * 100) if not drawdown.empty else 0

    sharpe_ratio = sortino_ratio = calmar_ratio = 0.0

    if len(daily_returns) > 1:
        days = (df.index[-1] - df.index[0]).days
        if days < 1:
            days = 1
            
        annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1
        
        annual_std_dev = daily_returns.std() * np.sqrt(365.25)
        sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON)
        
        downside_returns = daily_returns[daily_returns < 0]
        annual_downside_std = downside_returns.std() * np.sqrt(365.25)
        sortino_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_downside_std + EPSILON)
        
        calmar_ratio = (annual_return_rate * 100) / (max_drawdown + EPSILON)
    
    tradeBreakdown = []
    current_trade = {}
    for i, t in enumerate(trades):
        if t['action'] == 'buy' or t['action'] == 'sell_short':
            current_trade = {
                "entryTime": t['time'], "entryPrice": t['price'],
                "size": t['size'], "position": "long" if t['action'] == 'buy' else "short"
            }
        elif (t['action'] == 'sell' or t['action'] == 'cover') and 'entryTime' in current_trade:
            current_trade.update({
                "exitTime": t['time'], "exitPrice": t['price'],
                "profit": t['profit_usd'], "reason": t['reason'],
                "result": "win" if t['profit_usd'] > 0 else "loss" if t['profit_usd'] < 0 else "breakeven"
            })
            tradeBreakdown.append(current_trade)
            current_trade = {} 

    total_trades = len(tradeBreakdown); win_count = 0; lose_count = 0;
    gross_profit = 0.0; gross_loss = 0.0; avg_win = 0.0; avg_loss = 0.0;
    if total_trades > 0:
        win_pnl = [t['profit'] for t in tradeBreakdown if t['profit'] > 0]
        lose_pnl = [t['profit'] for t in tradeBreakdown if t['profit'] <= 0]
        win_count = len(win_pnl); lose_count = len(lose_pnl)
        gross_profit = sum(win_pnl); gross_loss = abs(sum(lose_pnl))
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        p_factor = gross_profit / (gross_loss + EPSILON)
        avg_win = sum(win_pnl) / (win_count + EPSILON)
        avg_loss = abs(sum(lose_pnl)) / (lose_count + EPSILON)
    else:
        win_rate = 0.0; p_factor = 0.0
    
    # --- 💡 START UPGRADE (v5) 💡 ---
    # This is the NEW fix for the 'DataFrame columns are not unique' warning
    
    # 1. Reset index
    candle_df = df.reset_index().rename(columns={'datetime':'timestamp'})
    
    # 2. De-duplicate the columns before selecting.
    # This keeps the *first* instance of 'open', 'high', 'low', 'close', 'volume'.
    candle_df = candle_df.loc[:, ~candle_df.columns.duplicated(keep='first')]
    
    # 3. Define the *only* columns we want.
    candle_data_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # 4. Check which of these *actually* exist
    final_candle_cols = [col for col in candle_data_cols if col in candle_df.columns]
    
    # 5. Create a *new, clean DataFrame* with only the unique columns
    # This DataFrame is now guaranteed to have no duplicates.
    candle_df_clean = candle_df[final_candle_cols]
    
    # 6. Call .to_dict() on the *clean* DataFrame. This will not warn.
   
    if 'timestamp' in candle_df_clean.columns:
       candle_df_clean['timestamp'] = candle_df_clean['timestamp'].apply(lambda x: x.isoformat())

    candle_data_dict = candle_df_clean.to_dict('records')
    # --- 💡 END OF FIX 💡 ---
    
    return {
        "metrics": {
            "totalReturn": total_return, "profitFactor": p_factor, "maxDrawdown": max_drawdown, 
            "winRate": win_rate, "totalTrades": total_trades, "averageWin": avg_win, "averageLoss": avg_loss, 
            "finalBalance": final_balance, "winningTrades": win_count, "losingTrades": lose_count,
            "sharpeRatio": sharpe_ratio, "sortinoRatio": sortino_ratio, "calmarRatio": calmar_ratio
        },
        "equity": equity_curve,
        "tradeBreakdown": tradeBreakdown,
        "candleData": candle_data_dict  # Use the fixed dictionary
    }
