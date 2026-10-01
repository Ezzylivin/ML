import os
import traceback
import logging
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import asyncio
from datetime import datetime, timedelta

# Internal Project Imports
from .manager import PrecisionPyramidManager
from .strategies import compute_signal, aggregate_signals
from .utils import recursive_clean

logger = logging.getLogger("BacktestService")

async def execute_backtest(
    symbol: str = 'BTC-USD', 
    timeframe: str = '1h', 
    startDate: str = "2025-01-01",
    endDate: str = "2026-01-01",
    initial_balance: float = 1000.0,
    risk_percentage: float = 1.0, 
    strategies: list = [],
    **kwargs
) -> dict:
    """
    ENGINE v100.2: The Self-Healing Backtester
    Fixes: Fuzzy BB Mapping, Column Normalization, and Optimizer API Safety.
    """
    try:
        # --- 🟢 STEP 1: HARDENED DATA LOAD ---
        file_path = f"data/{symbol}-{timeframe}.csv"
        df = None

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # CCXT FALLBACK: Automated download
            logger.info(f"🌐 [DATA] Local Missing. Fetching {symbol} via CCXT...")
            clean_symbol = symbol.replace('-', '/')
            exchange = ccxt.coinbase({'timeout': 60000, 'enableRateLimit': True})
            start_dt = datetime.strptime(startDate, "%Y-%m-%d")
            since_ts = int((start_dt - timedelta(hours=300)).timestamp() * 1000)
            
            batch = await asyncio.to_thread(exchange.fetch_ohlcv, clean_symbol, timeframe, since=since_ts, limit=2000)
            if batch:
                df = pd.DataFrame(batch, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)

        if df is None or df.empty:
            raise ValueError(f"CRITICAL: No data source found for {symbol}")

        # Normalize Time Column
        rename_map = {c: 'time' for c in df.columns if c.lower() in ['time', 'timestamp', 'datetime', 'date']}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.info(f"✅ Normalized columns: {rename_map}")

        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True, drop=False)
        
        # --- 🟢 STEP 2: FUZZY INDICATOR MAPPING ---
        params = kwargs.get('params', {})
        
        # Base Indicators
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=200, append=True, col_names=("sma_trend",))
        
        # Bollinger Bands with Fuzzy Mapping
        bbands = df.ta.bbands(length=20, std=2)
        if bbands is not None and not bbands.empty:
            # Find columns containing BBU (Upper) and BBL (Lower)
            bbu_cols = [c for c in bbands.columns if 'BBU' in c]
            bbl_cols = [c for c in bbands.columns if 'BBL' in c]
            
            if bbu_cols and bbl_cols:
                df['BBU'] = bbands[bbu_cols[0]]
                df['BBL'] = bbands[bbl_cols[0]]
            else:
                raise KeyError("Could not find BBU/BBL in calculated bands.")
        
        df.fillna(0, inplace=True)

        # --- 🟢 STEP 3: INITIALIZATION ---
        req_start = pd.to_datetime(startDate).tz_localize('UTC')
        req_end = pd.to_datetime(endDate).tz_localize('UTC')
        df_trading = df.loc[req_start:req_end]
        
        if df_trading.empty:
            return {"metrics": {"roi": -50.0}, "status": "no_data"}

        mgr = PrecisionPyramidManager(capital=initial_balance, symbol=symbol, base_risk=risk_percentage/100, **params)

        # --- 🟢 STEP 4: SIMULATION LOOP ---
        for i in range(1, len(df_trading)):
            current_time = df_trading.index[i]
            current_row = df_trading.iloc[i]
            time_str = current_time.isoformat()
            df_slice = df.loc[:current_time] 

            # UI Filters
            s_h, e_h = int(params.get('session_start', 12)), int(params.get('session_end', 21))
            is_prime = s_h <= current_time.hour <= e_h if s_h <= e_h else (current_time.hour >= s_h or current_time.hour <= e_h)

            strategy_votes = [compute_signal(df_slice, s) for s in strategies]
            decision_data = aggregate_signals(
                strategy_votes, 
                df_slice=df_slice, 
                adx_val=current_row.get('ADX_14', 0), 
                **params
            )

            # Execution Gate
            if decision_data['decision'] == 'EXIT' and mgr.positions:
                mgr._close_position(current_row['close'], time_str, "Momentum Stall")
                continue

            if mgr.positions:
                mgr.check_exit(current_row['high'], current_row['low'], time_str)
            
            if not mgr.positions and decision_data['decision'] in ['LONG', 'SHORT'] and is_prime:
                active_side = 1 if decision_data['decision'] == 'LONG' else -1
                mgr.handle(
                    signal=active_side, price=current_row['open'], time=time_str, 
                    atr=current_row.get('ATRr_14', 0), adx=current_row.get('ADX_14', 0), 
                    tslAtrMult=params.get('tslAtrMult', 3.0)
                )
            
            mgr.step_equity(current_row['close'], time_str)

        return recursive_clean(mgr.get_results())

    except Exception as e:
        logger.error(f"❌ Backtest Crash: {traceback.format_exc()}")
        # Optimizer Safety Net: Always return a valid ROI key
        return {"metrics": {"roi": -50.0, "totalTrades": 0}, "error": str(e), "status": "failed"}
