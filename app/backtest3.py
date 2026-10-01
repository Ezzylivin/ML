import os
import sys 
import asyncio
import traceback
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
from datetime import datetime

# Internal Project Imports
from app.manager2 import PrecisionPyramidManager
from app.utils2 import recursive_clean
from app.predictors.model_factory import ModelFactory 

# Hardened Environment Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

logger = logging.getLogger("BacktestService")

FEATURES_10 = [
    'open', 'high', 'low', 'close', 'rsi', 
    'atr', 'adx', 'adx_logic', 'atr_logic', 'sma_logic'
]

async def execute_backtest(
    symbol: str = 'BTC-USD', 
    timeframe: str = '1h', 
    startDate: str = "2024-12-01",
    endDate: str = "2026-01-02",
    initialBalance: float = 1000.0, 
    risk_percentage: float = 1.0, 
    model_type: str = "XGBoost",
    params: dict = None,
    **kwargs 
) -> dict:
    try:
        # 🟢 1. SETUP & CONFIGURATION
        tf.keras.backend.clear_session()
        incoming_params = params if params is not None else {}
        
        # --- NEW: Control Parameters ---
        # 1. Direction Filter: 'LONG', 'SHORT', or 'BOTH'
        allowed_direction = incoming_params.get('trade_direction', 'BOTH').upper()
        
        # 2. Strategy Switching Threshold (ML Confidence)
        # 0.15 means: Prob > 0.65 is Trend, Prob < 0.35 is Trend. Everything else is Range.
        conf_threshold = float(incoming_params.get('ml_confidence_threshold', 0.15))
        
        # 3. Strategy Selection
        trend_strat = incoming_params.get('trend_strategy', 'atr_breakout')
        range_strat = incoming_params.get('range_strategy', 'bollinger_reversal')

        is_genetic = incoming_params.get('is_genetic_run', False)
        if not is_genetic:
            with open("engine_heartbeat.txt", "w") as f: f.write(f"{symbol} | 2% | Loading Data...")

        lookback = int(kwargs.get('lookback') or incoming_params.get('lookback', 50))
        
        # 🟢 2. DATA PREPARATION & INDICATORS
        file_path = f"data/{symbol}-{timeframe}.csv"
        if not os.path.exists(file_path): raise FileNotFoundError(f"Missing data file: {file_path}")
            
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # --- Standard Indicators ---
        df['rsi'] = df.ta.rsi(length=14)
        df['atr'] = df.ta.atr(length=14)
        adx_df = df.ta.adx(length=14)
        df['adx'] = adx_df.iloc[:, 0]
        df['sma_200'] = df.ta.sma(length=200) # Trend Baseline
        
        # --- Strategy-Specific Indicators ---
        # We calculate these so the strategies have data to work with even if ML is off
        df['sma_50'] = df.ta.sma(length=50)   # For Breakouts
        df.ta.bbands(length=20, std=2.0, append=True) # For Reversals (BBL_20_2.0, etc)

        # Map to "Logic" columns for feature consistency
        df['sma_logic'] = df['sma_200']
        df['adx_logic'] = df['adx']
        df['atr_logic'] = df['atr']
        df = df.fillna(0)
        
        # Temporal Slicing
        time_col = 'datetime' if 'datetime' in df.columns else next((c for c in df.columns if 'time' in c or 'date' in c), df.columns[0])
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()

        req_start = pd.to_datetime(startDate).tz_localize('UTC') if pd.to_datetime(startDate).tzinfo is None else pd.to_datetime(startDate)
        req_end = pd.to_datetime(endDate).tz_localize('UTC') if pd.to_datetime(endDate).tzinfo is None else pd.to_datetime(endDate)
        df_trading = df.loc[req_start:req_end].copy() 

        # 🟢 3. MODEL LOADING (With Safety Fallback)
        custom_signals = incoming_params.get('custom_signal')
        predictor = None
        use_ml = True # Assume we want ML unless proved otherwise

        # Check if user explicitly requested NO ML or passed "None" as model
        if model_type == "None" or incoming_params.get('use_ml') == False:
            use_ml = False
        
        if use_ml and custom_signals is None:
            predictor = kwargs.get('ml_model')
            if predictor is None:
                try:
                    predictor = ModelFactory.load_model(model_type, symbol)
                except Exception as e:
                    logger.warning(f"⚠️ ML Load Failed ({str(e)}). Switching to Pure Strategy Mode.")
                    use_ml = False

        # 🟢 4. MANAGER INITIALIZATION
        mgr = PrecisionPyramidManager(
            capital=initialBalance, symbol=symbol, base_risk=float(risk_percentage/100),
            commission=float(incoming_params.get('commission', 0.0006)), 
            slippage=float(incoming_params.get('slippage', 0.0001)),
            tslAtrMult=float(incoming_params.get('tslAtrMult', 3.0))
        )

        # 🟢 5. MAIN SIMULATION LOOP
        # Align genetic signals if present
        if custom_signals is not None:
            custom_signals = pd.Series(custom_signals, index=df.index).loc[req_start:req_end]

        for i in range(lookback, len(df_trading)): 
            current_time = df_trading.index[i]
            current_row = df_trading.iloc[i]

            # --- A. DETERMINE PROBABILITY / REGIME SOURCE ---
            prob = 0.50 # Default Neutral
            
            if custom_signals is not None:
                prob = float(custom_signals.iloc[i]) # Genetic Override
            elif use_ml and predictor is not None:
                # Ask the AI
                state = df.loc[:current_time].tail(lookback)[FEATURES_10]
                try:
                    prob = predictor.predict_direction(state) # 0.0 to 1.0
                except:
                    prob = 0.50 
            else:
                # === PURE STRATEGY FALLBACK ===
                # If ML is off, we simulate "confidence" using ADX
                # This ensures the strategy logic below still works!
                adx_threshold = incoming_params.get('regime_threshold', 25)
                if current_row['adx_logic'] > adx_threshold:
                    prob = 0.99 # Fake High Confidence -> Triggers Trend Mode
                else:
                    prob = 0.50 # Fake Neutral -> Triggers Range Mode

            # --- B. DETERMINE REGIME ---
            confidence = abs(prob - 0.50)
            is_trend_regime = confidence > conf_threshold 

            # --- C. GENERATE RAW SIGNAL ---
            raw_signal = 0 # 0=Hold, 1=Buy, -1=Sell

            if is_trend_regime:
                # 🦁 TREND LOGIC (High Confidence / High ADX)
                if trend_strat == 'atr_breakout':
                    mult = incoming_params.get('trend_param_1', 3.0)
                    # Verify direction matches "Prob" direction (Long vs Short)
                    if prob > 0.5 and current_row['close'] > (current_row['sma_50'] + (current_row['atr'] * mult)):
                        raw_signal = 1
                    elif prob < 0.5 and current_row['close'] < (current_row['sma_50'] - (current_row['atr'] * mult)):
                        raw_signal = -1
                
                elif trend_strat == 'ema_cross':
                    if prob > 0.5 and current_row['sma_50'] > current_row['sma_200']: raw_signal = 1
                    elif prob < 0.5 and current_row['sma_50'] < current_row['sma_200']: raw_signal = -1

            else:
                # 🦀 RANGE LOGIC (Low Confidence / Low ADX)
                if range_strat == 'bollinger_reversal':
                    bbl = current_row.get('BBL_20_2.0', 0)
                    bbu = current_row.get('BBU_20_2.0', 0)
                    # Buy Dip / Sell Rip
                    if current_row['close'] < bbl: raw_signal = 1
                    elif current_row['close'] > bbu: raw_signal = -1

                elif range_strat == 'rsi_oversold':
                    if current_row['rsi'] < 30: raw_signal = 1
                    elif current_row['rsi'] > 70: raw_signal = -1

            # --- D. APPLY DIRECTION FILTER (The Gatekeeper) ---
            final_signal = 0
            if allowed_direction == 'BOTH':
                final_signal = raw_signal
            elif allowed_direction == 'LONG' and raw_signal == 1:
                final_signal = 1
            elif allowed_direction == 'SHORT' and raw_signal == -1:
                final_signal = -1
            
            # UI Update 
            if not is_genetic and (i % 100 == 0 or i == len(df_trading) - 1):
                progress_pct = round((i / len(df_trading)) * 100, 1)
                regime_str = 'TREND' if is_trend_regime else 'RANGE'
                with open("engine_heartbeat.txt", "w") as f:
                    f.write(f"{symbol} | {progress_pct}% | AI: {round(prob, 2)} | {regime_str}")
                await asyncio.sleep(0)

            # --- POSITION LIFECYCLE ---
            if mgr.positions:
                side = mgr.positions[0]['side']
                should_exit = False
                
                # Dynamic Exit Thresholds
                # Trend trades need room (0.40/0.60), Range trades need tight exits
                exit_threshold_long = 0.40 if is_trend_regime else 0.45
                exit_threshold_short = 0.60 if is_trend_regime else 0.55

                if custom_signals is not None:
                    if (side == 'long' and prob == -1) or (side == 'short' and prob == 1): should_exit = True
                elif use_ml:
                    # Only use ML confidence exit if ML is actually running
                    if (side == 'long' and prob < exit_threshold_long) or (side == 'short' and prob > exit_threshold_short):
                        should_exit = True

                if should_exit:
                    mgr._close_position(current_row['close'], current_time.isoformat(), "AI Conf Drop")
                else:
                    mgr.check_exit(current_row['high'], current_row['low'], current_time.isoformat())
            
            # --- ENTRY LOGIC ---
            if not mgr.positions:
                if final_signal == 1:
                    mgr.handle(1, current_row['open'], current_time.isoformat(), signal_prob=prob, atr=current_row['atr_logic'])
                elif final_signal == -1:
                    mgr.handle(-1, current_row['open'], current_time.isoformat(), signal_prob=prob, atr=current_row['atr_logic'])

            mgr.step_equity(current_row['close'], current_time.isoformat())

        # 🟢 6. FINALIZE
        if not is_genetic:
            with open("engine_heartbeat.txt", "w") as f: f.write(f"{symbol} | COMPLETE")
            
        return recursive_clean(mgr.get_results())

    except Exception as e:
        logger.error(f"❌ Backtest Crash: {traceback.format_exc()}")
        return {"status": "failed", "error": str(e), "metrics": {"roi": -100.0}}
